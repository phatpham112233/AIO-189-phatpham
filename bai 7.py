def valid_input(tp,fp,fn):
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    if tp <=0 or fp <= 0 or fn <= 0:
        print("tp and fp and fn must be greater than zero")
        return False
    
    return True
           
## precision and recall
def Precision_calc(tp, fp):
    Precision = tp / (tp+fp)
    return Precision
def Recall_calc(tp, fn):
    Recall = tp / (tp+fn)
    return Recall

##f1 score
def F1score_calc(Precision,Recall):
    if Precision + Recall == 0:
        return 0
    PRtimesRC = (Precision*Recall)
    PRplusRC = (Precision+Recall)
    F1score = 2 * (PRtimesRC) / (PRplusRC)
    return F1score



##input of tp fp fn
try:
    tp = (input("please input number for tb: "))
    fp = (input("please input number for fp: "))
    fn = (input("Please input number for fn: "))


##result of recall, precision, f1score
    if  valid_input(tp,fp,fn):
        result_Recall = Recall_calc(tp, fn)
        result_Precision = Precision_calc(tp, fp)
        result_F1score = F1score_calc(result_Precision,result_Recall)


##print

        print (f"precision value is {result_Precision}")
        print (f"recall value is {result_Recall}")
        print (f"F1-score value is {result_F1score}")

except ValueError:
    print("Ivalid input, please enter valid number and not letter to tp, tp, and fn, and ensure the number are greater than zero.")

