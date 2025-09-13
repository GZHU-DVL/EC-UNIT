import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numpy import argmax
from torchvision import utils
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
import math
import clip
from tqdm import tqdm

from model.generator  import Generator
from model.discriminator import Discriminator
from model.content_encoder import ContentEncoder
from model.vgg import VGGLoss
from model.arcface.id_loss import IDLoss
from PIL import Image

from util import load_image, visualize, adv_loss, r1_reg, divide_pred, moving_average
from dataset import create_unpaired_dataloader





class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train Adversarial Image Translation of EC-UNIT")
        self.parser.add_argument("--task", type=str, help="task type, e.g. cat2dog")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--iter", type=int, default=75000, help="iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch size")
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt',
                                 help="path to the saved content encoder")
        self.parser.add_argument("--generator_path", type=str, default='no',
                                 help="path to the saved content encoder")
        self.parser.add_argument("--identity_path", type=str, default='./checkpoint/model_ir_se50.pth',
                                 help="path to the identity model")
        self.parser.add_argument("--lambda_reconstruction", type=float, default=1.0,
                                 help="the weight of reconstruction loss")
        self.parser.add_argument("--lambda_mlcs", type=float, default=0.1, help="the weight of mlcs loss")
        self.parser.add_argument("--lambda_dfd", type=float, default=50.0, help="the weight of dfd loss")
        self.parser.add_argument("--lambda_mlsr", type=float, default=50.0, help="the weight of mlsr loss")
        self.parser.add_argument("--lambda_mlss", type=float, default=50.0, help="the weight of mlss loss")
        self.parser.add_argument("--lambda_mlcr", type=float, default=0.5, help="the weight of mlcr loss")
        self.parser.add_argument("--lambda_msk", type=float, default=1.0, help="the weight of msk loss")
        self.parser.add_argument("--lambda_cdd", type=float, default=0.05, help="the weight of cdd loss")
        self.parser.add_argument("--lambda_sdd", type=float, default=0.05, help="the weight of sdd loss")
        self.parser.add_argument("--lambda_vps", type=float, default=0.1, help="the weight of vps loss")
        self.parser.add_argument("--lambda_id", type=float, default=1.0, help="the weight of identity loss")
        self.parser.add_argument("--source_paths", type=str, nargs='+',
                                 help="the path to the training images in each source domain")
        self.parser.add_argument("--target_paths", type=str, nargs='+',
                                 help="the path to the training images in each target domain")
        self.parser.add_argument("--source_num", type=int, nargs='+', default=[0],
                                 help="the number of the training images in each source domain")
        self.parser.add_argument("--target_num", type=int, nargs='+', default=[0],
                                 help="the number of the training images in each target domain")
        self.parser.add_argument("--use_allskip", action="store_true",
                                 help="use dynamic skip connection to compute Lrec")
        self.parser.add_argument("--use_idloss", action="store_true", help="use identity loss")
        self.parser.add_argument("--not_flip_style", action="store_true",
                                 help="flip the style image to prevent learning pose of the style")
        self.parser.add_argument("--style_layer", type=int, default=4,
                                 help="the discriminator layer to extract style feature for Lsty")
        self.parser.add_argument("--save_every", type=int, default=5000, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=50000, help="when to start saving a checkpoint")
        self.parser.add_argument("--visualize_every", type=int, default=1000,
                                 help="interval of saving an intermediate result")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to the saved models")
        self.parser.add_argument("--mitigate_style_bias", action="store_true",
                                 help="mitigate style bias by use more rare styles when training sampler")
        self.parser.add_argument("--target_type", type=str, default=None, help="classification which breed or type, eg dog")
        self.parser.add_argument("--begin_iter", type=int, default=0,
                                 help="mitigate style bias by use more rare styles when training sampler")
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.source_num[0] == 0:
            self.opt.source_num = [int(1e8)] * len(self.opt.source_paths)
        if self.opt.target_num[0] == 0:
            self.opt.target_num = [int(1e8)] * len(self.opt.target_paths)
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt



def train(args, dataloader, netG, netD, optimizer_G, optimizer_D, netG_ema, vgg_loss, text_s, text_c, id_loss=None, device='cuda'):
    pbar = tqdm(range(args.iter), initial=0, smoothing=0.01, ncols=130, dynamic_ncols=False)

    netG.train()
    netD.train()
    netG_ema.eval()
    iterator = iter(dataloader)
    for idx in pbar:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            data = next(iterator)

        x, y = data['source'], data['target']
        x = x.to(device)
        y = y.to(device)


        with torch.no_grad():
            #print(x.shape)
            cfeat_x = netEC(x, get_feature=True)
            
            cfeat_y = netEC(y, get_feature=True)

        loss_dict = {}

        # flip style image to prevent learning pose of the style
        if args.not_flip_style or np.random.rand(1) < 0.5:
            y_ = y
        else:
            y_ = y[:, :, :, torch.arange(y.size(3) - 1, -1, -1).long()]

        # translation
        yhat, masks = netG(cfeat_x, y_)
        # reconstruction
        ybar, _ = netG(cfeat_y, y_, useskip=args.use_allskip)
        fake_and_real = torch.cat([yhat, y], dim=0)
        preds, sfeats = netD(fake_and_real, args.style_layer)
        fake_pred, real_pred = divide_pred(preds)
        Lgadv = adv_loss(fake_pred, 1)
        #mlcs
        hatcon=netEC(yhat,get_feature=True)
        Lmlcs=torch.stack([F.mse_loss(hatcon[i],cfeat_x[i]) for i in range(6)])
        Lmlcs=torch.sum(Lmlcs)*args.lambda_mlcs
        #dfd
        fake_style, real_style = divide_pred(sfeats)
        Ldfd = F.l1_loss(fake_style, real_style.detach()) * args.lambda_dfd
        #rec
        Lrec = (F.l1_loss(ybar, y) + vgg_loss(ybar, y)) * args.lambda_reconstruction
        Lid = torch.tensor(0.0, device=device)
        if args.use_idloss:
            Lid = id_loss(yhat, y) * args.lambda_id
        # vps
        Lvps = vgg_loss(yhat, y) * args.lambda_vps

        #mlss & mlsr
        yhat_sty = netG.style_encoder(yhat, True)
        yori_sty = netG.style_encoder(y, True)
        ybar_sty = netG.style_encoder(ybar,True)
        Lmlsr = torch.stack([F.mse_loss(ybar_sty[i], yori_sty[i]) for i in range(6)])
        Lmlsr = torch.sum(Lmlsr) * args.lambda_mlsr
        Lmlss = torch.stack([F.mse_loss(yhat_sty[i], yori_sty[i]) for i in range(6)])
        Lmlss = torch.sum(Lmlss) * args.lambda_mlss
        #Lmlcr
        Cbar=netEC(ybar,get_feature=True)
        Lmlcr = torch.stack([F.mse_loss(Cbar[i], cfeat_y[i]) for i in range(6)])
        Lmlcr = torch.sum(Lmlcr)*args.lambda_mlcr
        #record
        loss_dict['g'] = Lgadv
        loss_dict['mlcs'] = Lmlcs
        loss_dict['dfd'] = Ldfd
        loss_dict['rec'] = Lrec
        loss_dict['vps']=Lvps
        loss_dict['mlsr']=Lmlsr
        loss_dict['mlcr']=Lmlcr
        loss_dict['mlss']=Lmlss
        if args.use_idloss:
            loss_dict['id'] = Lid
        
        g_loss = Lgadv  + Lrec + Lid +Lmlss+Lmlcs+ Lmlcr+Lmlsr+ Ldfd+ Lvps
        # print("1")
        #if(idx+1>(args.iter/2)):
        if(idx+args.begin_iter+1>0):
            
            
            Lmsk = torch.tensor(0.0, device=device)
            for mask in masks:
                Lmsk += torch.mean(mask) * args.lambda_msk
            loss_dict['msk'] = Lmsk
            g_loss =g_loss + Lmsk

            with torch.no_grad():
                c_xori=F.interpolate(x,size=(224,224),mode='bilinear')
                s_yori=F.interpolate(y,size=(224,224),mode='bilinear')
                s_yhat=F.interpolate(yhat,size=(224,224),mode='bilinear')
                s_ybar=F.interpolate(ybar,size=(224,224),mode='bilinear')
                image_features_yori = cmodel.encode_image(s_yori)
                image_features_yhat = cmodel.encode_image(s_yhat)
                image_features_ybar = cmodel.encode_image(s_ybar)
                image_features_xori = cmodel.encode_image(c_xori)
                
                n_image_features_xori =image_features_xori/ image_features_xori.norm(dim=-1, keepdim=True)
                n_image_features_yori =image_features_yori/ image_features_yori.norm(dim=-1, keepdim=True)
                n_image_features_yhat =image_features_yhat/ image_features_yhat.norm(dim=-1, keepdim=True)
                n_image_features_ybar =image_features_ybar/ image_features_ybar.norm(dim=-1, keepdim=True)
                CrossEloss = nn.CrossEntropyLoss()
        
            text_s_features = cmodel.encode_text(text_s)
            n_text_s_features = text_s_features/text_s_features.norm(dim=-1, keepdim=True)

            similarity_yori = (100.0 * n_image_features_yori @ n_text_s_features.T)

            similarity_yhat = (100.0 * n_image_features_yhat @ n_text_s_features.T)
            similarity_ybar = (100.0 * n_image_features_ybar @ n_text_s_features.T)

            
            Lsdd = (CrossEloss(similarity_yhat ,similarity_yori.softmax(dim=-1))
                    +CrossEloss(similarity_ybar ,similarity_yori.softmax(dim=-1))
                    +CrossEloss(similarity_yhat ,similarity_ybar.softmax(dim=-1)))*args.lambda_sdd
            loss_dict['sdd'] = Lsdd

            g_loss =g_loss + Lsdd
        
            text_c_features_list = []
            for key, value in text_c.items():
                # print(key,value)
                text_c_features = cmodel.encode_text(value)
                n_text_c_features = text_c_features/text_c_features.norm(dim=-1, keepdim=True)
                text_c_features_list.append(n_text_c_features)
            # print(text_c_features)
            Lcdd = 0
            for text_c_features in text_c_features_list:
                c_similarity_xori = (100.0 * n_image_features_xori @ text_c_features.T)
                c_similarity_yori = (100.0 * n_image_features_yori @ text_c_features.T)
                c_similarity_yhat = (100.0 * n_image_features_yhat @ text_c_features.T)
                c_similarity_ybar = (100.0 * n_image_features_ybar @ text_c_features.T)
                Lcdd += (CrossEloss(c_similarity_yhat, c_similarity_xori.softmax(dim=-1))
                    + CrossEloss(c_similarity_ybar, c_similarity_yori.softmax(dim=-1)))* args.lambda_cdd
            Lcdd /= len(text_c_features_list)
            loss_dict['cdd'] = Lcdd
            g_loss =g_loss +Lcdd

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        with torch.no_grad():
            yhat, _ = netG(cfeat_x, y_)

        y.requires_grad_()
        fake_and_real = torch.cat([yhat.detach(), y], dim=0)
        preds, _ = netD(fake_and_real)
        fake_pred, real_pred = divide_pred(preds)

        Ldadv = adv_loss(real_pred, 1) + adv_loss(fake_pred, 0)
        Lr1 = r1_reg(real_pred, y)

        d_loss = Ldadv + Lr1

        loss_dict['d'] = Ldadv
        loss_dict['r1'] = Lr1

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        moving_average(netG, netG_ema, beta=0.999)
        message = ''
        for k, v in loss_dict.items():
            v = v.mean().float()
            message += 'L%s: %.3f ' % (k, v)
        tqdm.write("\033[A \033[A")
        tqdm.write(message)

        if ((idx + 1+args.begin_iter) >= args.save_begin and (idx + 1+args.begin_iter) % args.save_every == 0) or (idx + 1+args.begin_iter) == args.iter:
            torch.save(
                {
                    "g": netG.state_dict(),
                    "d": netD.state_dict(),
                    "g_ema": netG_ema.state_dict(),
                    "g_optim": optimizer_G.state_dict(),
                    "d_optim": optimizer_D.state_dict(),
                    # "args": args,
                },
                f"%s/%s-%05d.pt" % (args.model_path, args.task, idx + 1+args.begin_iter),
            )

        if (idx+args.begin_iter) == 0 or (idx + 1+args.begin_iter) % args.visualize_every == 0 or (idx + 1+args.begin_iter) == args.iter:
            with torch.no_grad():
                yhat2, _ = netG_ema(cfeat_x, y_)

            viznum = min(args.batch, 8)
            sample = F.adaptive_avg_pool2d(torch.cat((x[0:viznum].cpu(), y[0:viznum].cpu(),
                                                      yhat[0:viznum].cpu(), yhat2[0:viznum].cpu()), dim=0), 128)
            utils.save_image(
                sample,
                f"log/%s/%05d.jpg" % (args.task, (idx + 1+args.begin_iter)),
                nrow=viznum,
                normalize=True,
            )



if __name__ == "__main__":

    parser = TrainOptions()
    args = parser.parse()
    print('*' * 98)

    if not os.path.exists("log/%s/" % (args.task)):
        os.makedirs("log/%s/" % (args.task))

    device = 'cuda'
    netEC = ContentEncoder()
    netEC.load_state_dict(torch.load(args.content_encoder_path, map_location=lambda storage, loc: storage),strict=False)
    netEC = netEC.to(device)
    for param in netEC.parameters():
        param.requires_grad = False

    device=torch.device("cuda"if torch.cuda.is_available()else "cpu")
    cmodel, preprocess = clip.load("ViT-B/32", device=device)

    # prepare text list for SPM
    # ----------------------------------------
    if args.target_type == None:
        raise Exception("Please write in your target type")
    if args.target_type == 'dog':
        text_s = clip.tokenize(["Affenpinscher dog breed","Afghan_hound dog breed","African_hunting_dog dog breed","Aldale dog breed","American_staffordshire_terrier dog breed","Appenzeller dog breed","Australian_terrier dog breed","Basenji dog breed","Basset dog breed","Beagle dog breed","Bedlington_terrier dog breed","Bernese_mountain_dog dog breed","Black-and-tan_coonhound dog breed","Blenheim_spaniel dog breed","Bloodhound dog breed","Bluetick dog breed","Border_collie dog breed","Border_terrier dog breed","Borzoi dog breed","Boston_bull dog breed","Bouvier_des_flandres dog breed","Boxer dog breed","Brabancon_griffon dog breed","Briard dog breed","Brittany_spaniel dog breed","Bull_mastiff dog breed","Cairn dog breed","Cardigan dog breed","Chesapeake_bay_retriever dog breed","Chihuahua dog breed","Chow dog breed","Clumber dog breed","Cocker_spaniel dog breed","Collie dog breed","Curlycoated_retriever dog breed","Dandie_dinmont dog breed","Dhole dog breed","Dingo dog breed","Doberman dog breed","English_foxhound dog breed","English_setter dog breed","English_springer dog breed","Entlebucher dog breed","Eskimo_dog dog breed","Flat-coated_retriever dog breed","French_bulldog dog breed","German_shepherd dog breed","German_short-haired_pointer dog breed","Giant_schnauzer dog breed","Golden_retriever dog breed","Gordon_setter dog breed","Great_dane dog breed","Great_pyrenees dog breed","Greater_swiss_mountain_dog dog breed","Groenendael dog breed","Ibizan_hound dog breed","Irish_setter dog breed","Irish_terrier dog breed","Irish_water_spaniel dog breed","Irish_wolfhound dog breed","Italian_greyhound dog breed","Japanese_spaniel dog breed","Keeshond dog breed","Kelpie dog breed","Kerry_blue_terrier dog breed","Komondor dog breed","Kuvasz dog breed","Labrador_retriever dog breed","Lakeland_terrier dog breed","Leonberg dog breed","Lhasa dog breed","Malamute dog breed","Malinois dog breed","Maltese_dog dog breed","Mexican_hairless dog breed","Miniature_pinscher dog breed","Miniature_poodle dog breed","Miniature_schnauzer dog breed","Newfoundland dog breed","Norfolk_terrier dog breed","Norwegian_elkhound dog breed","Norwich_terrier dog breed","Old_english_sheepdog dog breed","Otterhound dog breed","Papillon dog breed","Pekinese dog breed","Pembroke dog breed","Pomeranian dog breed","Pug dog breed","Redbone dog breed","Rhodesian_ridgeback dog breed","Rottweiler dog breed","Saint_bernard dog breed","Saluki dog breed","Samoyed dog breed","Schipperke dog breed","Scotch_terrier dog breed","Scottish_deerhound dog breed","Sealyham_terrier dog breed","Shetland_sheepdog dog breed","Shih-tzu dog breed","Siberian_husky dog breed","Silky_terrier dog breed","Soft-coated_wheaten_terrier dog breed","Staffordshire_bullterrier dog breed","Standard_poodle dog breed","Standard_schnauzer dog breed","Sussex_spaniel dog breed","Tibetan_mastiff dog breed","Tibetan_terrier dog breed","Toy_poodle dog breed","Toy_terrier dog breed","Vizsla dog breed","Walker_hound dog breed","Weimaraner dog breed","Welsh_springer_spaniel dog breed","West_highland_white_terrier dog breed","Whippet dog breed","Wire-haired_fox_terrier dog breed","Yorkshire_terrier dog breed"]).to(device)
        text_c_dict  = {
            'text_c1' : ["facing left", "facing right", "facing forward"],
            'text_c2' : ["portrait with a side view", "portrait with a frontal view", "portrait with a three-quarter view", "bust shot", "full body shot"],
            'text_c3' : ["with an open mouth", "with a closed mouth", "others"]
        }
        # 将描述词汇组合成一个变量 text_c
        text_c = {key: clip.tokenize(descriptors).to(device) for key, descriptors in text_c_dict.items()}
    # ----------------------------------------
    if args.target_type == 'cat':
        text_s = clip.tokenize(["Abyssinian cat breed", "American Bobtail cat breed", "American Curl cat breed", "American Shorthair cat breed", "American Wirehair cat breed", "Applehead Siamese cat breed", "Balinese cat breed", "Bengal cat breed", "Birman cat breed" ,"Bombay cat breed", "British Shorthair cat breed", "Burmese cat breed", "Burmilla cat breed", "Calico cat breed", "Canadian Hairless cat breed", "Chartreux cat breed", "Chausie cat breed", "Chinchilla cat breed", "Cornish Rex cat breed", "Cymric cat breed", "Devon Rex cat breed", "Dilute Calico cat breed", "Dilute Tortoiseshell cat breed", "Domestic Long Hair cat breed", "Domestic Medium Hair cat breed", "Domestic Short Hair cat breed", "Egyptian Mau cat breed", "Exotic Shorthair cat breed", "Extra-Toes Cat - Hemingway Polydactyl cat breed",  "Havana cat breed",  "Himalayan cat breed", "Japanese Bobtail cat breed", "Javanese cat breed", "Korat cat breed", "LaPerm cat breed", "Maine Coon cat breed", "Manx cat breed", "Munchkin cat breed", "Nebelung cat breed", "Norwegian Forest Cat breed", "Ocicat cat breed", "Oriental Long Hair cat breed", "Oriental Short Hair cat breed", "Oriental Tabby cat breed", "Persian cat breed", "Pixiebob cat breed", "Ragamuffin cat breed", "Ragdoll cat breed", "Russian Blue cat breed", "Scottish Fold cat breed", "Selkirk Rex cat breed", "Siamese cat breed", "Siberian cat breed", "Silver cat breed", "Singapura cat breed", "Snowshoe cat breed", "Somali cat breed", "Sphynx - Hairless Cat breed", "Tabby cat breed", "Tiger cat breed", "Tonkinese cat breed", "Torbie cat breed", "Tortoiseshell cat breed", "Turkish Angora cat breed", "Turkish Van cat breed", "Tuxedo cat breed", "York Chocolate cat breed"]).to(device)
        text_c_dict = {
            'text_c1': ["facing left", "facing right", "facing forward"],
            'text_c2': ["portrait with a side view", "portrait with a frontal view", "portrait with a three-quarter view", "bust shot", "full body shot"],
            'text_c3': ["with an open mouth", "with a closed mouth", "others"]
        }
        # 将描述词汇组合成一个变量 text_c
        text_c = {key: clip.tokenize(descriptors).to(device) for key, descriptors in text_c_dict.items()}
    
    # ----------------------------------------
    if args.target_type == 'face':
         # text = clip.tokenize(["early childhood age stage", "teenager adolescence age stage", "early adulthood age stage", "midlife age stage", "mature adulthood age stage", "late adulthood age stage", "the elderly people"]).to(device)
         text_s = clip.tokenize(["a child", "a teenager", "a young adult", "middle-aged people", "late adulthood people", "the elderly people", "baldhead people","the people wear a hat", "the people wear glasses","profile face"]).to(device)
         #text1= clip.tokenize(["angry facial expression", "disgust facial expression", "fear facial expression", "happy facial expression", "neutral facial expression", "sad facial expression", "surprise facial expression"]).to(device)
         text_c_dict = {
             'text_c1' : ["side portrait", "frontal portrait", "three-quarter view of a person's face"],
             'text_c2' : ["angry face", "disgust face", "fear face", "happy face with open-mouthed smile", "happy face with closed-mouthed smile", "neutral face", "sad face", "surprise face"],
             'text_c3' : ["headshot", "headshot with a hat", "headshot with glasses", "headshot with earrings"]
         }
         # 将描述词汇组合成一个变量 text_c
         text_c = {key: clip.tokenize(descriptors).to(device) for key, descriptors in text_c_dict.items()}

    # ----------------------------------------
    if args.target_type == 'bird':
        text_s = clip.tokenize(["ABBOTTS BABBLER bird breed", "ABBOTTS BOOBY bird breed", "ABYSSINIAN GROUND HORNBILL bird breed", "AFRICAN CROWNED CRANE bird breed", "AFRICAN EMERALD CUCKOO bird breed", "AFRICAN FIREFINCH bird breed", "AFRICAN OYSTER CATCHER bird breed", "AFRICAN PIED HORNBILL bird breed", "AFRICAN PYGMY GOOSE bird breed", "ALBATROSS bird breed", "ALBERTS TOWHEE bird breed", "ALEXANDRINE PARAKEET bird breed", "ALPINE CHOUGH bird breed", "ALTAMIRA YELLOWTHROAT bird breed", "AMERICAN AVOCET bird breed", "AMERICAN BITTERN bird breed", "AMERICAN COOT bird breed", "AMERICAN FLAMINGO bird breed", "AMERICAN GOLDFINCH bird breed", "AMERICAN KESTREL bird breed", "AMERICAN PIPIT bird breed", "AMERICAN REDSTART bird breed", "AMERICAN ROBIN bird breed", "AMERICAN WIGEON bird breed", "AMETHYST WOODSTAR bird breed", "ANDEAN GOOSE bird breed", "ANDEAN LAPWING bird breed", "ANDEAN SISKIN bird breed", "ANHINGA bird breed", "ANIANIAU bird breed", "ANNAS HUMMINGBIRD bird breed", "ANTBIRD bird breed", "ANTILLEAN EUPHONIA bird breed", "APAPANE bird breed", "APOSTLEBIRD bird breed", "ARARIPE MANAKIN bird breed", "ASHY STORM PETREL bird breed", "ASHY THRUSHBIRD bird breed", "ASIAN CRESTED IBIS bird breed", "ASIAN DOLLARD BIRD bird breed", "AUCKLAND SHAQ bird breed", "AUSTRAL CANASTERO bird breed", "AUSTRALASIAN FIGBIRD bird breed", "AVADAVAT bird breed", "AZARAS SPINETAIL bird breed", "AZURE BREASTED PITTA bird breed", "AZURE JAY bird breed", "AZURE TANAGER bird breed", "AZURE TIT bird breed", "BAIKAL TEAL bird breed", "BALD EAGLE bird breed", "BALD IBIS bird breed", "BALI STARLING bird breed", "BALTIMORE ORIOLE bird breed", "BANANAQUIT bird breed", "BAND TAILED GUAN bird breed", "BANDED BROADBILL bird breed", "BANDED PITA bird breed", "BANDED STILT bird breed", "BAR-TAILED GODWIT bird breed", "BARN OWL bird breed", "BARN SWALLOW bird breed", "BARRED PUFFBIRD bird breed", "BARROWS GOLDENEYE bird breed", "BAY-BREASTED WARBLER bird breed", "BEARDED BARBET bird breed", "BEARDED BELLBIRD bird breed", "BEARDED REEDLING bird breed", "BELTED KINGFISHER bird breed", "BIRD OF PARADISE bird breed", "BLACK AND YELLOW BROADBILL bird breed", "BLACK BAZA bird breed", "BLACK COCKATO bird breed", "BLACK FACED SPOONBILL bird breed", "BLACK FRANCOLIN bird breed", "BLACK HEADED CAIQUE bird breed", "BLACK NECKED STILT bird breed", "BLACK SKIMMER bird breed", "BLACK SWAN bird breed", "BLACK TAIL CRAKE bird breed", "BLACK THROATED BUSHTIT bird breed", "BLACK THROATED HUET bird breed", "BLACK THROATED WARBLER bird breed", "BLACK VENTED SHEARWATER bird breed", "BLACK VULTURE bird breed", "BLACK-CAPPED CHICKADEE bird breed", "BLACK-NECKED GREBE bird breed", "BLACK-THROATED SPARROW bird breed", "BLACKBURNIAM WARBLER bird breed", "BLONDE CRESTED WOODPECKER bird breed", "BLOOD PHEASANT bird breed", "BLUE COAU bird breed", "BLUE DACNIS bird breed", "BLUE GRAY GNATCATCHER bird breed", "BLUE GROSBEAK bird breed", "BLUE GROUSE bird breed", "BLUE HERON bird breed", "BLUE MALKOHA bird breed", "BLUE THROATED TOUCANET bird breed", "BOBOLINK bird breed", "BORNEAN BRISTLEHEAD bird breed", "BORNEAN LEAFBIRD bird breed", "BORNEAN PHEASANT bird breed", "BRANDT CORMARANT bird breed", "BREWERS BLACKBIRD bird breed", "BROWN CREPPER bird breed", "BROWN HEADED COWBIRD bird breed", "BROWN NOODY bird breed", "BROWN THRASHER bird breed", "BUFFLEHEAD bird breed", "BULWERS PHEASANT bird breed", "BURCHELLS COURSER bird breed", "BUSH TURKEY bird breed", "CAATINGA CACHOLOTE bird breed", "CACTUS WREN bird breed", "CALIFORNIA CONDOR bird breed", "CALIFORNIA GULL bird breed", "CALIFORNIA QUAIL bird breed", "CAMPO FLICKER bird breed", "CANARY bird breed", "CANVASBACK bird breed", "CAPE GLOSSY STARLING bird breed", "CAPE LONGCLAW bird breed", "CAPE MAY WARBLER bird breed", "CAPE ROCK THRUSH bird breed", "CAPPED HERON bird breed", "CAPUCHINBIRD bird breed", "CARMINE BEE-EATER bird breed", "CASPIAN TERN bird breed", "CASSOWARY bird breed", "CEDAR WAXWING bird breed", "CERULEAN WARBLER bird breed", "CHARA DE COLLAR bird breed", "CHATTERING LORY bird breed", "CHESTNET BELLIED EUPHONIA bird breed", "CHINESE BAMBOO PARTRIDGE bird breed", "CHINESE POND HERON bird breed", "CHIPPING SPARROW bird breed", "CHUCAO TAPACULO bird breed", "CHUKAR PARTRIDGE bird breed", "CINNAMON ATTILA bird breed", "CINNAMON FLYCATCHER bird breed", "CINNAMON TEAL bird breed", "CLARKS GREBE bird breed", "CLARKS NUTCRACKER bird breed", "COCK OF THE  ROCK bird breed", "COCKATOO bird breed", "COLLARED ARACARI bird breed", "COLLARED CRESCENTCHEST bird breed", "COMMON FIRECREST bird breed", "COMMON GRACKLE bird breed", "COMMON HOUSE MARTIN bird breed", "COMMON IORA bird breed", "COMMON LOON bird breed", "COMMON POORWILL bird breed", "COMMON STARLINGbird breed", "COPPERY TAILED COUCAL bird breed", "CRAB PLOVER bird breed", "CRANE HAWK bird breed", "CREAM COLORED WOODPECKER bird breed", "CRESTED AUKLET bird breed", "CRESTED CARACARA bird breed", "CRESTED COUA bird breed", "CRESTED FIREBACK bird breed", "CRESTED KINGFISHER bird breed", "CRESTED NUTHATCH bird breed", "CRESTED OROPENDOLA bird breed", "CRESTED SERPENT EAGLE bird breed", "CRESTED SHRIKETIT bird breed", "CRESTED WOOD PARTRIDGE bird breed", "CRIMSON CHAT bird breed", "CRIMSON SUNBIRD bird breed", "CROW bird breed", "CROWNED PIGEON bird breed", "CUBAN TODY bird breed", "CUBAN TROGON bird breed", "CURL CRESTED ARACURI bird breed",
                              "D-ARNAUDS BARBET bird breed", "DALMATIAN PELICAN bird breed", "DARJEELING WOODPECKER bird breed", "DARK EYED JUNCO bird breed", "DAURIAN REDSTART bird breed", "DEMOISELLE CRANE bird breed", "DOUBLE BARRED FINCH bird breed", "DOUBLE BRESTED CORMARANT bird breed", "DOUBLE EYED FIG PARROT bird breed", "DOWNY WOODPECKER bird breed", "DUSKY LORY bird breed", "DUSKY ROBIN bird breed", "EARED PITA bird breed", "EASTERN BLUEBIRD bird breed", "EASTERN BLUEBONNET bird breed", "EASTERN GOLDEN WEAVER bird breed", "EASTERN MEADOWLARK bird breed", "EASTERN ROSELLA bird breed", "EASTERN TOWEE bird breed", "EASTERN WIP POOR WILL bird breed", "EASTERN YELLOW ROBIN bird breed", "ECUADORIAN HILLSTAR bird breed", "EGYPTIAN GOOSE bird breed", "ELEGANT TROGON bird breed", "ELLIOTS  PHEASANT bird breed", "EMERALD TANAGER bird breed", "EMPEROR PENGUIN bird breed", "EMU bird breed", "ENGGANO MYNA bird breed", "EURASIAN BULLFINCH bird breed", "EURASIAN GOLDEN ORIOLE bird breed", "EURASIAN MAGPIE bird breed", "EUROPEAN GOLDFINCH bird breed", "EUROPEAN TURTLE DOVE bird breed", "EVENING GROSBEAK bird breed", "FAIRY BLUEBIRD bird breed", "FAIRY PENGUIN bird breed", "FAIRY TERN bird breed", "FAN TAILED WIDOW bird breed", "FASCIATED WREN bird breed", "FIERY MINIVET bird breed", "FIORDLAND PENGUIN bird breed", "FIRE TAILLED MYZORNIS bird breed", "FLAME BOWERBIRD bird breed", "FLAME TANAGER bird breed", "FRIGATE bird breed", "FRILL BACK PIGEON bird breed", "GAMBELS QUAIL bird breed", "GANG GANG COCKATOO bird breed", "GILA WOODPECKER bird breed", "GILDED FLICKER bird breed", "GLOSSY IBIS bird breed", "GO AWAY BIRD bird breed", "GOLD WING WARBLER bird breed", "GOLDEN BOWER BIRD bird breed", "GOLDEN CHEEKED WARBLER bird breed", "GOLDEN CHLOROPHONIA bird breed", "GOLDEN EAGLE bird breed", "GOLDEN PARAKEET bird breed", "GOLDEN PHEASANT bird breed", "GOLDEN PIPIT bird breed", "GOULDIAN FINCH bird breed", "GRANDALA bird breed", "GRAY CATBIRD bird breed", "GRAY KINGBIRD bird breed", "GRAY PARTRIDGE bird breed", "GREAT ARGUS bird breed", "GREAT GRAY OWL bird breed", "GREAT JACAMAR bird breed", "GREAT KISKADEE bird breed", "GREAT POTOO bird breed", "GREAT TINAMOU bird breed", "GREAT XENOPS bird breed", "GREATER PEWEE bird breed", "GREATER PRAIRIE CHICKEN bird breed", "GREATOR SAGE GROUSE bird breed", "GREEN BROADBILL bird breed", "GREEN JAY bird breed", "GREEN MAGPIE bird breed", "GREEN WINGED DOVE bird breed", "GREY CUCKOOSHRIKE bird breed", "GREY HEADED FISH EAGLE bird breed", "GREY PLOVER bird breed", "GROVED BILLED ANI bird breed", "GUINEA TURACO bird breed", "GUINEAFOWL bird breed", "GURNEYS PITTA bird breed", "GYRFALCON bird breed", "HAMERKOP bird breed", "HARLEQUIN DUCK bird breed", "HARLEQUIN QUAIL bird breed", "HARPY EAGLE bird breed", "HAWAIIAN GOOSE bird breed", "HAWFINCH bird breed", "HELMET VANGA bird breed", "HEPATIC TANAGER bird breed", "HIMALAYAN BLUETAIL bird breed", "HIMALAYAN MONAL bird breed", "HOATZIN bird breed", "HOODED MERGANSER bird breed", "HOOPOES bird breed", "HORNED GUAN bird breed", "HORNED LARK bird breed", "HORNED SUNGEM bird breed", "HOUSE FINCH bird breed", "HOUSE SPARROW bird breed", "HYACINTH MACAW bird breed", "IBERIAN MAGPIE bird breed", "IBISBILL bird breed", "IMPERIAL SHAQ bird breed", "INCA TERN bird breed", "INDIAN BUSTARD bird breed", "INDIAN PITTA bird breed", "INDIAN ROLLER brd breed", " bird breed", "INDIAN VULTURE bird breed", "INDIGO BUNTING bird breed", "INDIGO FLYCATCHER bird breed", "INLAND DOTTEREL bird breed", "IVORY BILLED ARACARI bird breed", "IVORY GULL bird breed", "IWI bird breed", "JABIRU bird breed", "JACK SNIPE bird breed", "JACOBIN PIGEON bird breed", "JANDAYA PARAKEET bird breed", "JAPANESE ROBIN bird breed", "JAVA SPARROW bird breed", "JOCOTOCO ANTPITTA bird breed", "KAGU bird breed", "KAKAPO bird breed", "KILLDEAR bird breed", "KING EIDER bird breed", "KING VULTURE bird breed", "KIWI bird breed", "KOOKABURRA bid breed", " bird breed", "LARK BUNTING bird breed", "LAUGHING GULL bird breed", "LAZULI BUNTING bird breed", "LESSER ADJUTANT bird breed", "LILAC ROLLER bird breed", "LIMPKIN bird breed", "LITTLE AUK bird breed", "LOGGERHEAD SHRIKE bird breed", "LONG-EARED OWL bird breed", "LOONEY BIRDS bird breed", "LUCIFER HUMMINGBIRD bird breed", "MAGPIE GOOSE bird breed", "MALABAR HORNBILL bird breed", "MALACHITE KINGFISHER bird breed", "MALAGASY WHITE EYE bird breed", "MALEO bird breed", "MALLARD DUCK bird breed", "MANDRIN DUCK bird breed", "MANGROVE CUCKOO bird breed", "MARABOU STORK bird breed", "MASKED BOBWHITE bird breed", "MASKED BOOBY bird breed", "MASKED LAPWING bird breed", "MCKAYS BUNTING bird breed", "MERLIN bird breed", "MIKADO  PHEASANT bird breed", "MILITARY MACAW bird breed", "MOURNING DOVE bird breed", "MYNA bird breed", "NICOBAR PIGEON bird breed", "NOISY FRIARBIRD bird breed", "NORTHERN BEARDLESS TYRANNULET bird breed", "NORTHERN CARDINAL bird breed", "NORTHERN FLICKER bird breed", "NORTHERN FULMAR bird breed", "NORTHERN GANNET bird breed", "NORTHERN GOSHAWK bird breed", "NORTHERN JACANA bird breed", "NORTHERN MOCKINGBIRD bird breed", "NORTHERN PARULA bird breed", "NORTHERN RED BISHOP bird breed", "NORTHERN SHOVELER bird breed", "OCELLATED TURKEY bird breed", "OKINAWA RAIL bird breed", "ORANGE BRESTED BUNTING bird breed", "ORIENTAL BAY OWL bird breed", "ORNATE HAWK EAGLE bird breed", "OSPREY bird breed", "OSTRICH bird breed",
                              "OVENBIRD bird breed", "OYSTER CATCHER bird breed", "PAINTED BUNTING bird breed", "PALILA  bird breed", "PALM NUT VULTURE bird breed", "PARADISE TANAGER bird breed", "PARAKETT  AKULET bird breed", "PARUS MAJOR bird breed", "PATAGONIAN SIERRA FINCH bird breed", "PEACOCK bird breed", "PEREGRINE FALCON bird breed", "PHAINOPEPLA bird breed", "PHILIPPINE EAGLE bird breed", "PINK ROBIN bird breed", "PLUSH CRESTED JAY bird breed", "POMARINE JAEGER bird breed", "PUFFIN bird breed", "PUNA TEAL bird breed", "PURPLE FINCH bird breed", "PURPLE GALLINULE bird breed", "PURPLE MARTIN bird breed", "PURPLE SWAMPHEN bird breed", "PYGMY KINGFISHER bird breed", "PYRRHULOXIA bird breed", "QUETZAL bird breed", "RAINBOW LORIKEET bird breed", "RAZORBILL bird breed", "RED BEARDED BEE EATER bird breed", "RED BELLIED PITTA bird breed", "RED BILLED TROPICBIRD bird breed", "RED BROWED FINCH bird breed", "RED FACED CORMORANT bird breed", "RED FACED WARBLER bird breed", "RED FODY bird breed", "RED HEADED DUCK bird breed", "RED HEADED WOODPECKER bird breed", "RED KNOT bird breed", "RED LEGGED HONEYCREEPER bird breed", "RED NAPED TROGON bird breed", "RED SHOULDERED HAWK bird breed", "RED TAILED HAWK bird breed", "RED TAILED THRUSH bird breed", "RED WINGED BLACKBIRD bird breed", "RED WISKERED BULBUL bird breed", "REGENT BOWERBIRD bird breed", "RING-NECKED PHEASANT bird breed", "ROADRUNNER bird breed", "ROCK DOVE bird breed", "ROSE BREASTED COCKATOO bird breed", "ROSE BREASTED GROSBEAK bird breed", "ROSEATE SPOONBILL bird breed", "ROSY FACED LOVEBIRD bird breed", "ROUGH LEG BUZZARD bird breed", "ROYAL FLYCATCHER bird breed", "RUBY CROWNED KINGLET bird breed", "RUBY THROATED HUMMINGBIRD bird breed", "RUDY KINGFISHER bird breed", "RUFOUS KINGFISHER bird breed", "RUFUOS MOTMOT bird breed", "SAMATRAN THRUSH bird breed", "SAND MARTIN bird breed", "SANDHILL CRANE bird breed", "SATYR TRAGOPAN bird breed", "SAYS PHOEBE bird breed", "SCARLET CROWNED FRUIT DOVE bird breed", "SCARLET FACED LIOCICHLA bird breed", "SCARLET IBIS bird breed", "SCARLET MACAW bird breed", "SCARLET TANAGER bird breed", "SHOEBILL bird breed", "SHORT BILLED DOWITCHER bird breed", "SMITHS LONGSPUR bird breed", "SNOW GOOSE bird breed", "SNOWY EGRET bird breed", "SNOWY OWL bird breed", "SNOWY PLOVER bird breed", "SORA bird breed", "SPANGLED COTINGA bird breed", "SPLENDID WREN bird breed", "SPOON BILED SANDPIPER bird breed", "SPOTTED CATBIRD bird breed", "SPOTTED WHISTLING DUCK bird breed", "SRI LANKA BLUE MAGPIE bird breed", "STEAMER DUCK bird breed", "STORK BILLED KINGFISHER bird breed", "STRIATED CARACARA bird breed", "STRIPED OWL bird breed", "STRIPPED MANAKIN bird breed", "STRIPPED SWALLOW bird breed", "SUNBITTERN bird breed", "SUPERB STARLING bird breed", "SURF SCOTER bird breed", "SWINHOES PHEASANT bird breed", "TAILORBIRD bird breed", "TAIWAN MAGPIE bird breed", "TAKAHE bird breed", "TASMANIAN HEN bird breed", "TAWNY FROGMOUTH bird breed", "TEAL DUCK bird breed", "TIT MOUSE bird breed", "TOUCHAN bird breed", "TOWNSENDS WARBLER bird breed", "TREE SWALLOW bird breed", "TRICOLORED BLACKBIRD  bird breed", "TROPICAL KINGBIRD bird breed", "TRUMPTER SWAN bird breed", "TURKEY VULTURE bird breed", "TURQUOISE MOTMOT bird breed", "UMBRELLA BIRD bird breed", "VARIED THRUSH bird breed", "VEERY bird breed", "VENEZUELIAN TROUPIAL bird breed", "VERDIN bird breed", "VERMILION FLYCATHER bird breed", "VICTORIA CROWNED PIGEON bird breed", "VIOLET BACKED STARLING bird breed", "VIOLET GREEN SWALLOW bird breed", "VIOLET TURACO bird breed", "VULTURINE GUINEAFOWL bird breed", "WALL CREAPER bird breed", "WATTLED CURASSOW bird breed", "WATTLED LAPWING bird breed", "WHIMBREL bird breed", "WHITE BROWED CRAKE bird breed", "WHITE CHEEKED TURACO bird breed", "WHITE CRESTED HORNBILL bird breed", "WHITE EARED HUMMINGBIRD bird breed", "WHITE NECKED RAVEN bird breed", "WHITE TAILED TROPIC bird breed", "WHITE THROATED BEE EATER bird breed", "WILD TURKEY bird breed", "WILLOW PTARMIGAN bird breed", "WILSONS BIRD OF PARADISE bird breed", "WOOD DUCK bird breed", "WOOD THRUSH bird breed", "WRENTIT bird breed", "YELLOW BELLIED FLOWERPECKER bird breed", "YELLOW CACIQUE bird breed", "YELLOW HEADED BLACKBIRD bird breed", "RUFOUS TREPE bird breed", "ASIAN GREEN BEE EATER bird breed", "RUDDY SHELDUCK bird breed", "COPPERSMITH BARBET bird breed", "FOREST WAGTAIL bird breed", "WHITE BREASTED WATERHEN bird breed", "WOODLAND KINGFISHER bird breed", "SQUACCO HERON bird breed", "VISAYAN HORNBILL bird breed", "ZEBRA DOVE bird breed"]).to(device)
        text_c_dict = {
            'text_c1' : ["facing left", "facing right", "facing forward"],
            'text_c2' : ["portrait with a side view", "portrait with a frontal view", "portrait with a three-quarter view", "bust shot", "full body shot"]
        }
        # 将描述词汇组合成一个变量 text_c
        text_c = {key: clip.tokenize(descriptors).to(device) for key, descriptors in text_c_dict.items()}

    if args.target_type == 'car':
        text_s = clip.tokenize(["ambulance", "bus", "double-decker bus", "caravan", "van", "coach", "jeep", "convertible", "sedan", "formula car", "fire engine", "pickup truck", "microcar", "limousine", "suv", "police car", "tow truck", "snow remover", "hatchback", "station wagon", "recreational vehicle", "sports car", "tractor trailer"]).to(device)
        text_c_dict = {
            'text_c1' : ["facing left", "facing right", "facing forward"],
            'text_c2' : ["portrait with a side view", "portrait with a frontal view", "portrait with a three-quarter view", "bust shot", "full body shot"]
        }
        # 将描述词汇组合成一个变量 text_c
        text_c = {key: clip.tokenize(descriptors).to(device) for key, descriptors in text_c_dict.items()}
    print("length_of_style is:"+str(len(text_s)))
    print("Available GPU count:", torch.cuda.device_count())
    netG = Generator()
    netG = netG
    netG=netG.to(device)
    netG_ema = Generator()
    netG_ema = netG_ema
    netG_ema=netG_ema.to(device)
    netD = Discriminator().to(device)
    netD = netD
    netD=netD.to(device)

    if (args.generator_path == 'no'):
        netG.init_weights('kaiming', 0.02)
        netD.init_weights('kaiming', 0.02)
        netG_ema = copy.deepcopy(netG)
        for param in netG_ema.parameters():
            param.requires_grad = False

        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    else:
        ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
        netG_ema.load_state_dict(ckpt['g_ema'])
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
        optimizer_G.load_state_dict(ckpt['g_optim'])
        optimizer_D.load_state_dict(ckpt['d_optim'])
    print('Create models successfully!')

    # for image translation
    dataloader = create_unpaired_dataloader(args.source_paths, args.target_paths,
                                            args.source_num, args.target_num, args.batch)


    print('Create dataloaders successfully!')

    vgg_loss = VGGLoss()
    vgg_loss.vgg = vgg_loss.vgg.to(device)
    if args.use_idloss:
        id_loss = IDLoss(args.identity_path).to(device).eval()
    else:
        id_loss = None


    train(args, dataloader, netG, netD, optimizer_G, optimizer_D, netG_ema,vgg_loss, text_s, text_c, id_loss, device)
