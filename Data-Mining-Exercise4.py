import keyboard
while True:
    print("_____________________________Tamrin #4 _________________________________\n\n")
    
    MySentence = "farshad,nematpour,farshad,nematpour,farshad"
    print("Our words are: ",MySentence)
    input("enter for Spliting: ")

    MySentence_Splited = MySentence.split(",")
    print("Words after Spliting : ", MySentence_Splited)
    print("Words after Split + Removing Duplicated values: " ,set(MySentence_Splited) )

    print("\n___________________________Farshad NematPour ___________________________\n\n")
    keyboard.wait('q')
    keyboard.send('ctrl+6')
