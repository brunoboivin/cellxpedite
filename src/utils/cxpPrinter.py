from __future__ import division
import datetime

cxpSignature = "[CELLXPEDITE]"

def cxpPrint(s):
    now = str(datetime.datetime.now())
    print(cxpSignature + '[' + now + '] ' + str(s))
