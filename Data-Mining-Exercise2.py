print("________________\n\n")\

mySentence = input("Enter a Sentence or Word: ")
str2 = ""

for i in range(len(mySentence)):
    if(i % 2 == 0):
        str2 = str2 + mySentence[i]
print("Original String :  ", mySentence)
print("Final String :     ", str2)

print("\n______End of Programm_____Farshad_Nematpour_AzadUniversityOfMashhad")