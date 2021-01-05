print("________________\n\n")\

Sentence = input("Enter a string: ")
Alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

print("Your sentence has ",len(Sentence),"letter tottaly.\n")
for i in range(0,26):
    print("______")
    print(Alphabet[i]," is: " ,Sentence.count(Alphabet[i]))
print("______End of Programm_____Farshad_Nematpour_AzadUniversityOfMashhad")
