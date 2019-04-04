import json

import random as rnd
import numpy as np
#import pandas as pd
import pystan as ps
import pickle
from hashlib import md5
import math
#from scipy.optimize import minimize

from flask import Flask
from flask import jsonify
application = Flask(__name__)

@application.route("/<string:name>/")
def roulette_bias(name):

    print(name)
    
    wheel = ['00','27','10','25','29','12','8','19','31','18','6','21','33',
         '16','4','23','35','14','2','0','28','9','26','30','11','7','20',
         '32','17','5','22','34','15','3','24','36','13','1']
    redblack = ['green', 
                'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red', 
                'black', 'red', 'black','red', 'black', 'red', 'black', 'red', 'black', 
                'green', 
                'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 
                'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black', 'red']
    evenodd = ['none', 
               'odd', 'even', 'odd', 'odd', 'even', 'even', 'odd', 'odd', 'even', 
               'even', 'odd', 'odd', 'even', 'even', 'odd', 'odd', 'even', 'even', 
               'none', 
               'even', 'odd', 'even', 'even', 'odd', 'odd', 'even', 'even', 'odd', 
               'odd', 'even', 'even', 'odd', 'odd', 'even', 'even', 'odd', 'odd']
    lowhigh = ['none',
              'high','low','high','high','low','low','high','high','low',
               'low','high','high','low','low','high','high','low','low',
              'none',
              'high','low','high','high','low','low','high','high','low',
               'low','high','high','low','low','high','high','low','low']
    dozens = ['none',
             3,1,3,3,1,1,2,3,2,1,2,3,2,1,2,3,2,1,
             'none',
             3,1,3,3,1,1,2,3,2,1,2,3,2,1,2,3,2,1]
    columns = ['none',
              3,1,1,2,3,2,1,1,3,3,3,3,1,1,2,2,2,2,
              'none',
              1,3,2,3,2,1,2,2,2,2,1,1,3,3,3,3,1,1]
    sixline = ['none',
              ('8','9'),('3','4'),('8','9'),('9','10'),('3','4'),('2','3'),('6','7'),('10','11'),('5','6'),
              ('1','2'),('6','7'),('10','11'),('5','6'),('1','2'),('7','8'),['11'],('4','5'),['1'],
              'none',
              ('9','10'),('2','3'),('8','9'),('9','10'),('3','4'),('2','3'),('6','7'),('10','11'),('5','6'),
              ('1','2'),('7','8'),['11'],('4','5'),['1'],('7','8'),['11'],('4','5'),['1']]
    square = ['none',
             ('16','18'),('5','7'),('15','17'),('17','18','19','20'),('6','8'),('3','4','5','6'),('11','13'),('19','21'),('10','12'),
             ('2','4'),('12','14'),('20','22'),('9','11'),('1','3'),('13','14','15','16'),('21','22'),('7','8','9','10'),('1','2'),
             'none',
             ('17','19'),('4','6'),('15','16','17','18'),('18','20'),('5','6','7','8'),('3','5'),('11','12','13','14'),('19','20','21','22'),('9','10','11','12'),
             ('1','2','3','4'),('13','15'),['21'],('8','10'),['2'],('14','16'),['22'],('7','9'),['1']]
    street = ['none',
             '9','4','9','10','4','3','7','11','6','2','7','11','6','2','8','12','5','1',
             'none',
             '10','3','9','10','4','3','7','11','6','2','8','12','5','1','8','12','5','1']
    split = ['none',
            ('40','42','45'),('13','16','18'),('38','41','43'),('44','46','47','49'),('15','17','20'),('9','11','12','14'),('28','31','33'),('48','51','53'),('25','27','30'),
            ('5','7','10'),('30','32','35'),('50','52','55'),('23','26','28'),('3','6','8'),('34','36','37','39'),('54','56','57'),('19','21','22','24'),('1','2','4'),
            'none',
            ('43','46','48'),('10','12','15'),('39','41','42','44'),('45','47','50'),('14','16','17','19'),('8','11','13'),('29','31','32','34'),('49','51','52','54'),('24','26','27','29'),
            ('4','6','7','9'),('33','36','38'),('53','56'),('20','22','25'),('2','5'),('35','37','40'),('55','57'),('18','21','23'),('1','3')]
    
    neighbor = []
    wheel2 = ['13','1','00','27','10','25','29','12','8','19','31','18','6','21','33',
             '16','4','23','35','14','2','0','28','9','26','30','11','7','20',
             '32','17','5','22','34','15','3','24','36','13','1','00','27']
    for i in range(len(wheel)):
        n = wheel[i]
        bet = wheel2[((i+2)-2):((i+2)+3)]
        neighbor.append(bet)
    
    
    # alpha = np.ones(38)
    # alpha[10] = alpha[10]-(alpha[10]/2.0)
    # alpha[11] = alpha[11]+alpha[10]
    # dir_prob = np.random.dirichlet(alpha, 1)
    # dir_prob = dir_prob.tolist()[0]
    
    
    # In[1656]:
    
    
    # unbiased test
    #dir_prob = np.ones(38)/38
    
    # single test
    # dir_prob = np.ones(38)/43
    # dir_prob[wheel.index('13')] += 1-sum(dir_prob)
    
    # neighbor test
    # dir_prob = np.ones(38)/43
    # dir_prob[wheel.index('13')] = 2/43
    # dir_prob[wheel.index('1')] = 2/43
    # dir_prob[wheel.index('00')] = 2/43
    # dir_prob[wheel.index('27')] = 2/43
    # dir_prob[wheel.index('10')] = 2/43
    
    # street test
    # dir_prob = np.ones(38)/44
    # dir_prob[wheel.index('16')] = 2/44
    # dir_prob[wheel.index('17')] = 2/44
    # dir_prob[wheel.index('18')] = 2/44
    # dir_prob[wheel.index('34')] = 2/44
    # dir_prob[wheel.index('35')] = 2/44
    # dir_prob[wheel.index('36')] = 2/44
    
    # sixline test
    # dir_prob = np.ones(38)/44
    # dir_prob[wheel.index('16')] = 2/44
    # dir_prob[wheel.index('17')] = 2/44
    # dir_prob[wheel.index('18')] = 2/44
    # dir_prob[wheel.index('19')] = 2/44
    # dir_prob[wheel.index('20')] = 2/44
    # dir_prob[wheel.index('21')] = 2/44
    
    # square test
    # dir_prob = np.ones(38)/42
    # dir_prob[wheel.index('8')] = 2/42
    # dir_prob[wheel.index('9')] = 2/42
    # dir_prob[wheel.index('11')] = 2/42
    # dir_prob[wheel.index('12')] = 2/42
    
    # col color test
    # dir_prob = np.ones(38)/50
    # dir_prob[wheel.index('3')] = 2/50
    # dir_prob[wheel.index('6')] = 2/50
    # dir_prob[wheel.index('9')] = 2/50
    # dir_prob[wheel.index('12')] = 2/50
    # dir_prob[wheel.index('15')] = 2/50
    # dir_prob[wheel.index('18')] = 2/50
    # dir_prob[wheel.index('21')] = 2/50
    # dir_prob[wheel.index('24')] = 2/50
    # dir_prob[wheel.index('27')] = 2/50
    # dir_prob[wheel.index('30')] = 2/50
    # dir_prob[wheel.index('33')] = 2/50
    # dir_prob[wheel.index('36')] = 2/50
    
    # sum(dir_prob)
    
    # spins = [wheel[x] for x in np.random.choice(38, size=5000, p=dir_prob)]
    spins = name.split(',')
    
    print(spins)
    print(len(spins))
    
    spins_count = {x : 0 for x in wheel}
    
    for i in spins:
        spins_count[i] += 1
    
    counts = np.array(list(spins_count.values()))
    
    model = """
    data {
      int<lower=1> m;
      int counts[m];
      vector<lower=0>[m] priors;
    }
    
    parameters {
      simplex[m] p;
    }
    
    model {
      p ~ dirichlet(priors);
      counts ~ multinomial(p);
    }
    """
    
    def StanModel_cache2(model_code, model_name=None, **kwargs):
        """Use just as you would `stan`"""
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
        try:
            sm = pickle.load(open(cache_fn, 'rb'))
        except:
            sm = ps.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f:
                pickle.dump(sm, f)
        else:
            print("Using cached StanModel")
        return sm

    def StanModel_cache(model_code, model_name=None, **kwargs):
        print("===== loading pickle file")
        pk = open('cached-stanmodelx.pkl', 'rb')
        print("===== deserializing")
        sm = pickle.load(pk)
        print("===== done")
        return sm
    
    
    # stanmodel = ps.StanModel(model_code=model)
    stanmodel = StanModel_cache2(model_code=model, model_name='stanmodel')
    data = {'m':38, 'counts':counts, 'priors':np.ones(38)}
    print("===== before sampling")
    fit = stanmodel.sampling(data=data)
    print("===== after sampling")
    fit_par = fit.extract(permuted=True)
    fit_par['p'].shape
    fit_samples = fit_par['p']
    
    def getRedBlackProb(p):
        redblack_dic = {'red':0, 'black':0, 'green':0}
        zipped_array = zip(wheel,p,redblack)
        for i in zipped_array:
            if i[2] == 'green':
                redblack_dic['green'] += i[1]
            elif i[2] == 'red':
                redblack_dic['red'] += i[1]
            else:
                redblack_dic['black'] += i[1]
                
        return redblack_dic
    
    def getEvenOddProb(p):
        evenodd_dic = {'even':0, 'odd':0, 'none':0}
        zipped_array = zip(wheel,p,evenodd)
        for i in zipped_array:
            if i[2] == 'even':
                evenodd_dic['even'] += i[1]
            elif i[2] == 'odd':
                evenodd_dic['odd'] += i[1]
            else:
                evenodd_dic['none'] += i[1]
                
        return evenodd_dic
    
    def getLowHighProb(p):
        lowhigh_dic = {'low':0, 'high':0, 'none':0}
        zipped_array = zip(wheel,p,lowhigh)
        for i in zipped_array:
            if i[2] == 'low':
                lowhigh_dic['low'] += i[1]
            elif i[2] == 'high':
                lowhigh_dic['high'] += i[1]
            else:
                lowhigh_dic['none'] += i[1]
                
        return lowhigh_dic
    
    def getDozensProb(p):
        dozens_dic = {'1':0, '2':0, '3':0, 'none':0}
        zipped_array = zip(wheel,p,dozens)
        for i in zipped_array:
            if i[2] == 1:
                dozens_dic['1'] += i[1]
            elif i[2] == 2:
                dozens_dic['2'] += i[1]
            elif i[2] == 3:
                dozens_dic['3'] += i[1]
            else:
                dozens_dic['none'] += i[1]
                
        return dozens_dic
    
    def getColumnsProb(p):
        columns_dic = {'1':0, '2':0, '3':0, 'none':0}
        zipped_array = zip(wheel,p,columns)
        for i in zipped_array:
            if i[2] == 1:
                columns_dic['1'] += i[1]
            elif i[2] == 2:
                columns_dic['2'] += i[1]
            elif i[2] == 3:
                columns_dic['3'] += i[1]
            else:
                columns_dic['none'] += i[1]
                
        return columns_dic
    
    def getSixlineProb(p):
        sixline_dic = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, 'none':0}
        zipped_array = zip(wheel,p,sixline)
        for i in zipped_array:
            if i[2] == 'none':
                sixline_dic['none'] += i[1]
            else:
                for j in range(len(i[2])):
                    sixline_dic[i[2][j]] += i[1]
                    
        return sixline_dic
    
    def getSquareProb(p):
        square_dic = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, 
                      '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0, 'none':0}
        zipped_array = zip(wheel,p,square)
        for i in zipped_array:
            if i[2] == 'none':
                square_dic['none'] += i[1]
            else:
                for j in range(len(i[2])):
                    square_dic[i[2][j]] += i[1]
                    
        return square_dic
    
    def getStreetProb(p):
        street_dic = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, 'none':0}
        zipped_array = zip(wheel,p,street)
        for i in zipped_array:
            if i[2] == 'none':
                street_dic['none'] += i[1]
            else:
                street_dic[i[2]] += i[1]
                    
        return street_dic
    
    def getSplitProb(p):
        split_dic = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, 
                     '12':0, '13':0, '14':0, '15':0, '16':0, '17':0, '18':0, '19':0, '20':0, '21':0, '22':0,
                     '23':0, '24':0, '25':0, '26':0, '27':0, '28':0, '29':0, '30':0, '31':0, '32':0, '33':0,
                     '34':0, '35':0, '36':0, '37':0, '38':0, '39':0, '40':0, '41':0, '42':0, '43':0, '44':0,
                     '45':0, '46':0, '47':0, '48':0, '49':0, '50':0, '51':0, '52':0, '53':0, '54':0, '55':0,
                     '56':0, '57':0, 'none':0}
        zipped_array = zip(wheel,p,split)
        for i in zipped_array:
            if i[2] == 'none':
                split_dic['none'] += i[1]
            else:
                for j in range(len(i[2])):
                    split_dic[i[2][j]] += i[1]
                    
        return split_dic
    
    def getNeighborProb(p):
        neighbor_dic = {x:0 for x in wheel}
        zipped_array = zip(wheel,p,neighbor)
        for i in zipped_array:
            for j in i[2]:
                neighbor_dic[j] += i[1]
                    
        return neighbor_dic
    
    
    single_prob = np.mean(fit_samples, axis=0)
    
    split_prob = np.mean([list(getSplitProb(x).values()) for x in fit_samples], axis=0)
    
    street_prob = np.mean([list(getStreetProb(x).values()) for x in fit_samples], axis=0)
    
    square_prob = np.mean([list(getSquareProb(x).values()) for x in fit_samples], axis=0)
    
    sixline_prob = np.mean([list(getSixlineProb(x).values()) for x in fit_samples], axis=0)
    
    columns_prob = np.mean([list(getColumnsProb(x).values()) for x in fit_samples], axis=0)
    
    dozens_prob = np.mean([list(getDozensProb(x).values()) for x in fit_samples], axis=0)
    
    lowhigh_prob = np.mean([list(getLowHighProb(x).values()) for x in fit_samples], axis=0)
    
    evenodd_prob = np.mean([list(getEvenOddProb(x).values()) for x in fit_samples], axis=0)
    
    redblack_prob = np.mean([list(getRedBlackProb(x).values()) for x in fit_samples], axis=0)
    
    neighbor_prob = np.mean([list(getNeighborProb(x).values()) for x in fit_samples], axis=0)
    
    single_pay = np.ones(38)*35
    split_pay = np.ones(57)*17
    street_pay = np.ones(12)*11
    square_pay = np.ones(22)*8
    sixline_pay = np.ones(11)*5
    column_pay = np.ones(3)*2
    dozen_pay = np.ones(3)*2
    lowhigh_pay = np.ones(2)*1
    evenodd_pay = np.ones(2)*1
    redblack_pay = np.ones(2)*1
    neighbor_pay = np.ones(38)*(35/5)
    
    single_cover = np.ones(38)*1
    split_cover = np.ones(57)*2
    street_cover = np.ones(12)*3
    square_cover = np.ones(22)*4
    sixline_cover = np.ones(11)*6
    column_cover = np.ones(3)*12
    dozen_cover = np.ones(3)*12
    lowhigh_cover = np.ones(2)*18
    evenodd_cover = np.ones(2)*18
    redblack_cover = np.ones(2)*18
    neighbor_cover = np.ones(38)*5
    
    single_expected = single_prob * single_pay + (1-single_prob) * -1
    split_expected = split_prob[:-1] * split_pay + (1-split_prob[:-1]) * -1
    street_expected = street_prob[:-1] * street_pay + (1-street_prob[:-1]) * -1
    square_expected = square_prob[:-1] * square_pay + (1-square_prob[:-1]) * -1
    sixline_expected = sixline_prob[:-1] * sixline_pay + (1-sixline_prob[:-1]) * -1
    column_expected = columns_prob[:-1] * column_pay + (1-columns_prob[:-1]) * -1
    dozen_expected = dozens_prob[:-1] * dozen_pay + (1-dozens_prob[:-1]) * -1
    lowhigh_expected = lowhigh_prob[:-1] * lowhigh_pay + (1-lowhigh_prob[:-1]) * -1
    evenodd_expected = evenodd_prob[:-1] * evenodd_pay + (1-evenodd_prob[:-1]) * -1
    redblack_expected = redblack_prob[:-1] * redblack_pay + (1-redblack_prob[:-1]) * -1
    neighbor_expected = neighbor_prob * neighbor_pay + (1-neighbor_prob) * -1
    
    single_bet_dic = {
        i:'s:'+x for i,x in enumerate(wheel)
    }
    
    split_bet_dic = {
        38: 'sp:1-2', 39: 'sp:2-3', 40: 'sp:1-4', 41: 'sp:2-5', 42: 'sp:3-6', 
        43: 'sp:4-5', 44: 'sp:5-6', 45: 'sp:4-7', 46: 'sp:5-8', 47: 'sp:6-9', 
        48: 'sp:7-8', 49: 'sp:8-9', 50: 'sp:7-10', 51: 'sp:8-11', 52: 'sp:9-12', 
        53: 'sp:10-11', 54: 'sp:11-12', 55: 'sp:10-13', 56: 'sp:11-14', 57: 'sp:12-15', 
        58: 'sp:13-14', 59: 'sp:14-15', 60: 'sp:13-16', 61: 'sp:14-17', 62: 'sp:15-18', 
        63: 'sp:16-17', 64: 'sp:17-18', 65: 'sp:16-19', 66: 'sp:17-20', 67: 'sp:18-21', 
        68: 'sp:19-20', 69: 'sp:20-21', 70: 'sp:19-22', 71: 'sp:20-23', 72: 'sp:21-24', 
        73: 'sp:22-23', 74: 'sp:23-24', 75: 'sp:22-25', 76: 'sp:23-26', 77: 'sp:24-27', 
        78: 'sp:25-26', 79: 'sp:26-27', 80: 'sp:25-28', 81: 'sp:26-29', 82: 'sp:27-30', 
        83: 'sp:28-29', 84: 'sp:29-30', 85: 'sp:28-31', 86: 'sp:29-32', 87: 'sp:30-33', 
        88: 'sp:31-32', 89: 'sp:32-33', 90: 'sp:31-34', 91: 'sp:32-35', 92: 'sp:33-36', 
        93: 'sp:34-35', 94: 'sp:35-36'
    }   
    
    street_bet_dic = {
        95: 'st:1-2-3', 96: 'st:4-5-6', 97: 'st:7-8-9', 98: 'st:10-11-12', 99: 'st:13-14-15',
        100: 'st:16-17-18', 101: 'st:19-20-21', 102: 'st:22-23-24', 103: 'st:25-26-27', 104: 'st:28-29-30', 
        105: 'st:31-32-33', 106: 'st:34-35-36'
    }
    
    square_bet_dic = {
        107: 'sq:1/2/4/5', 108: 'sq:2/3/5/6', 109: 'sq:4/5/7/8', 110: 'sq:5/6/8/9', 111: 'sq:7/8/10/11', 
        112: 'sq:8/9/11/12', 113: 'sq:10/11/13/14', 114: 'sq:11/12/14/15', 115: 'sq:13/14/16/17', 116: 'sq:14/15/17/18',
        117: 'sq:16/17/19/20', 118: 'sq:17/18/20/21', 119: 'sq:19/20/22/23', 120: 'sq:20/21/23/24', 121: 'sq:22/23/25/26',
        122: 'sq:23/24/26/27', 123: 'sq:25/26/28/29', 124: 'sq:26/27/29/30', 125: 'sq:28/29/31/32', 126: 'sq:29/30/32/33',
        127: 'sq:31/32/34/35', 128: 'sq:32/33/35/36'
    }
    
    sixline_bet_dic = {
        129: 'sx:1/2/3/4/5/6', 130: 'sx:4/5/6/7/8/9', 131: 'sx:7/8/9/10/11/12', 132: 'sx:10/11/12/13/14/15', 133: 'sx:13/14/15/16/17/18', 134: 'sx:16/17/18/19/20/21', 
        135: 'sx:19/20/21/22/23/24', 136: 'sx:22/23/24/25/26/27', 137: 'sx:25/26/27/28/29/30', 138: 'sx:28/29/30/31/32/33', 139: 'sx:31/32/33/34/35/36'
    }
    
    three_bet_dic = {
        140: 'c:1', 141: 'c:2', 142: 'c:3',
        143: 'd:1', 144: 'd:2', 145: 'd:3'
    }
    
    two_bet_dic = {
        146: 'low', 147: 'high',
        148: 'even', 149: 'odd',
        150: 'red', 151: 'black'
    } 
    
    neighbor_bet_dic = {
        i+152:'n:{}'.format(x) for i,x in enumerate(wheel)
    }
    
    bet_dic = {**single_bet_dic, **split_bet_dic, **street_bet_dic, **square_bet_dic, **sixline_bet_dic, **three_bet_dic, **two_bet_dic, **neighbor_bet_dic}
    
    cover_dic = {x:np.array([],dtype=int) for x in bet_dic.values()}
    for i in range(len(bet_dic.keys())):
        bet_str = bet_dic[i]
        if bet_str == 'low':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(lowhigh) if x is 'low']]
        elif bet_str == 'high':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(lowhigh) if x is 'high']]
        elif bet_str == 'even':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(evenodd) if x is 'even']]
        elif bet_str == 'odd':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(evenodd) if x is 'odd']]
        elif bet_str == 'red':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(redblack) if x is 'red']]
        elif bet_str == 'black':
            cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(redblack) if x is 'black']]
        else:
            bet_name,bet_par = bet_str.split(":")
            if bet_name == 's':
                cover_dic[bet_str] = [bet_par]
            elif bet_name == 'n':
                cover_dic[bet_str] = neighbor[wheel.index(bet_par)]
            elif bet_name == 'sp':
                cover_dic[bet_str] = bet_par.split('-')
            elif bet_name == 'st':
                cover_dic[bet_str] = bet_par.split('-')
            elif bet_name == 'sq':
                cover_dic[bet_str] = bet_par.split('/')
            elif bet_name == 'sx':
                cover_dic[bet_str] = bet_par.split('/')
            elif bet_name == 'c':
                cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(columns) if x is int(bet_par)]]
            elif bet_name == 'd':
                cover_dic[bet_str] = [wheel[w] for w in [i for i,x in enumerate(columns) if x is int(bet_par)]]
            else:
                print('something is wrong')
                cover_dic[bet_str] = None
    
    def bet_coverage(bet):
        nums = cover_dic[bet]
        return len(np.unique(nums))/38.0
    
    def total_coverage(x):
        nums = np.array([])
        for i in range(len(x)):
            if x[i] > 1e-4:
                nums = np.append(nums, cover_dic[bet_dic[i]])
        return len(np.unique(nums))/38.0
    
    #bet_prob = np.hstack([single_prob,split_prob[:-1],street_prob[:-1],square_prob[:-1],sixline_prob[:-1],columns_prob[:-1],dozens_prob[:-1],lowhigh_prob[:-1],evenodd_prob[:-1],redblack_prob[:-1],neighbor_prob])
    bet_expected = np.hstack([single_expected,split_expected,street_expected,square_expected,sixline_expected,column_expected,dozen_expected,lowhigh_expected,evenodd_expected,redblack_expected,neighbor_expected])
    best_expected = bet_expected[bet_expected.argsort()[-10:][::-1]]
    #best_prob = [bet_prob[x] for x in bet_expected.argsort()[-10:][::-1]]
    best_bet = [bet_dic[x] for x in bet_expected.argsort()[-10:][::-1]]
    best_coverage = [bet_coverage(bet_dic[x]) for x in bet_expected.argsort()[-10:][::-1]]
    
    #final_bet = [x for x in zip(best_bet, best_expected, best_coverage)]
    final_bets = ['{},{:.2f}'.format(x[0],x[1]) for x in zip(best_bet, best_expected)]

    print("====final bet====")
    for i in final_bets:
      print(i)
    print("====final bet====")

    count = 0
    max_count = 10
    best_bet_s = []
    for i in bet_expected.argsort()[-len(bet_expected):][::-1]:
        if (count < max_count):
            b = bet_dic[i]
            if b.split(':')[0] == 's':
                best_bet_s.append('{},{:.2f}'.format(b,bet_expected[i]))
                count += 1

    count = 0
    max_count = 5
    best_bet_n = []
    for i in bet_expected.argsort()[-len(bet_expected):][::-1]:
        if (count < max_count):
            b = bet_dic[i]
            if b.split(':')[0] == 'n':
                best_bet_n.append('{},{:.2f}'.format(b,bet_expected[i]))
                count += 1

    count = 0
    max_count = 10
    best_bet_spst = []
    for i in bet_expected.argsort()[-len(bet_expected):][::-1]:
        if (count < max_count):
            b = bet_dic[i]
            if b.split(':')[0] == 'sp' or b.split(':')[0] == 'st':
                best_bet_spst.append('{},{:.2f}'.format(b,bet_expected[i]))
                count += 1

    count = 0
    max_count = 10
    best_bet_sqsx = []
    for i in bet_expected.argsort()[-len(bet_expected):][::-1]:
        if (count < max_count):
            b = bet_dic[i]
            if b.split(':')[0] == 'sq' or b.split(':')[0] == 'sx':
                best_bet_sqsx.append('{},{:.2f}'.format(b,bet_expected[i]))
                count += 1

    count = 0
    max_count = 10
    best_bet_ee = []
    for i in bet_expected.argsort()[-len(bet_expected):][::-1]:
        if (count < max_count):
            b = bet_dic[i]
            if b.split(':')[0] != 'sq' and b.split(':')[0] != 'sx' and b.split(':')[0] != 'sp' and b.split(':')[0] != 'st' and b.split(':')[0] != 'n' and b.split(':')[0] != 's':
                best_bet_ee.append('{},{:.2f}'.format(b,bet_expected[i]))
                count += 1

    final_dic = {'best =====':final_bets, 's =====':best_bet_s, 'n =====':best_bet_n, 'sqsx =====':best_bet_sqsx, 'spst =====':best_bet_spst, 'ee =====':best_bet_ee}

    return jsonify(final_dic)
    #return final_dic

if __name__ == "__main__":
    application.run(host="0.0.0.0")
