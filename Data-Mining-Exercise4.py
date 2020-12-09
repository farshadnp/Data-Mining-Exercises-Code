import keyboard
while True:
    print("_____________________________Tamrin #4 _________________________________\n\n")
    MySentence = "farshad,nematpour,25,Ahvaz,Mashhad,Azad-University,Data-Minning"
    print("Our words are: ",MySentence)
    input("enter for Spliting: ")

    MySentence_Splited = MySentence.split(",")
    print("Word after packaging : ", MySentence_Splited)

    print("\n___________________________Farshad NematPour ___________________________\n\n")
    keyboard.wait('q')
    keyboard.send('ctrl+6')