#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
import math


def getwords(doc):
    splitter=re.compile('\\W*')
    #単語を非アルファベットの文字で分解する
    words =[s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20 ]
    #ユニークな単語のみ取り出す"
    return dict([(w,1) for w in words])



def sampletrain(cl):
    cl.train("Nobody owns  the water ","good")
    cl.train("make quick rabbit jumps fences ", "good")
    cl.train("buy pharmaceuticals now","bad")
    cl.train("make quick money at the online casio", "bad")
    cl.train("the quick bronw fox jumps","good")


class classifier:
    def __init__(self,getfeatures,filename=None):
        #特徴量/カテゴリのカウント
        self.fc = {}

        #それぞれのカテゴリの中野ドキュメント数
        self.cc ={}
        self.getfeatures=getfeatures


    #特徴量/カテゴリのカウント数を増やす
    def incf(self,f,cat):
        self.fc.setdefault(f,{})
        self.fc[f].setdefault(cat,0)
        self.fc[f][cat]+=1


    #カテゴリのカウントを増やす
    def incc(self,cat):
        self.cc.setdefault(cat,0)
        self.cc[cat]+=1

    #あるカテゴリの中に特徴が現れた数
    def fcount(self,f,cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    #あるカテゴリのアイテムの数
    def catcount(self,cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0


    #アイテムたちの総数
    def totalcount(self):
        return sum(self.cc.values())

    #全てのカテゴリたちのリスト
    def categories(self):
        return self.cc.keys()

    def train(self,item,cat):
        features = self.getfeatures(item)

        #このカテゴリの中の特徴たちのカウントを増やす
        for f in features:
            self.incf(f,cat)
        #このカテゴリのカウントを増やす
        self.incc(cat)

    def fprob(self,f,cat):
        if self.catcount(cat)==0: return 0

        #このカテゴリ中に特徴量が出現する回数を,このカテゴリ中のアイテムの総数で割る
        return self.fcount(f,cat)/self.catcount(cat)


    def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
        #現在の確率を計算する
        basicprob =prf(f,cat)

        #この特徴量が全てのカテゴリ中に出現する数を数える
        totals = sum([self.fcount(f,c) for c in self.categories()])

        #重み付けした平均を計算
        bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp



class naivebayes(classifier):

    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.thresholds={}

    def setthreshold(self,cat,t):
        self.thresholds[cat] = t

    def getthreshold(self,cat):
        if cat not in self.thresholds:return 1.0
        return self.thresholds[cat]

    def classify(self,item,default=None):
        probs = {}
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item,cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        #確率が閾値*2番目にベストなものを超えているかを確認する
        for cat in probs:
            if cat == best:continue
            if probs[cat] * self.getthreshold(best) > probs[best]: return default
        return best





    def docprob(self,item,cat):
        features = self.getfeatures(item)
        #全ての特徴の確率を掛け合わせる
        p = 1
        for f in features: p *= self.weightedprob(f, cat, self.fprob)
        return p


    def prob(self,item,cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob



class fisherclassifier(classifier):
    def cprob(self,f,cat):
        #このカテゴリでこの特徴の頻度
        clf = self.fprob(f,cat)
        print "clf=",clf
        if clf == 0: return 0
        freqsum = sum([self.fprob(f,c) for  c in self.categories()])
        print "freqsum=",freqsum
        p = clf / (freqsum)
        return p

    def invchi(self,chi,df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1,df//2):
            term*=m/i
            sum+=term
        return min(sum,1.0)


    def fisherprob(self,item,cat):
        p = 1
        features = self.getfeatures(item)
        for  f in features:
            p*=(self.weightedprob(f, cat, self.cprob))
