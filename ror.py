dict2 = { 'where' : 4 ,
     'who' : 5 ,
     'why': 6 ,
     'this' : 20 
     }
list1, list2 = ['a', 'b', 'c'], [1,2,3]
dict2=dict( zip( list1, list2))
print (dict2)