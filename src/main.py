#!/usr/bin/env python
# -*- coding: utf-8 -*-


import docclass


cl = docclass.naivebayes(docclass.getwords)
docclass.sampletrain(cl)



print cl.classify("quick rabbit",default="unkown")
print cl.classify("quick money",default="unkown")

cl.setthreshold("bad", 3.0)
print cl.classify("quick money",default="unkown")

for i in range(10): docclass.sampletrain(cl)

print cl.classify("quick money",default="unkown")


cl = docclass.fisherclassifier(docclass.getwords)
docclass.sampletrain(cl)
print cl.cprob("quick","good")
print cl.cprob("money","bad")






