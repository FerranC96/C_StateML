Models trained using datasets with 8 different cell state markers:
1.  pRB
2.  IdU
3.  Cyclin B1
4.  pHH3
5.  cPARP
8.  cCaspase3
9.  PLK1
10. Geminin

Note missing entries in the list above due to the addition of 2 PTM markers (pAKT_T308 and pP38) found to have high feature importances.

[NOTE]
As of March 2022, the only model saved with the bigger set of 8 state antibodies (+ 2 ptms) is that trained with one Marias earliest datasets, in a negative control where PDO21 (human) was left untreated. 

Tested against the same PDO but treated with high concentrations of SN38 it seemed to hold up quite respectably. 
Still remains to be seen how it would perform against a murine dataset.