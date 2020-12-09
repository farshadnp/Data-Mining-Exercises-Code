import keyboard
while True:
    print("_____________________________Tamrin #3 _________________________________\n\n")

    MySentence = input("Please enter a long sentence : ")
    MySentence_Splited = MySentence.split(" ")


    print("\nBe soorat List kalamate jomle vared shode : ",MySentence_Splited)
    print("Words totall: ",len(MySentence_Splited))


    print("\n___________________________Farshad NematPour ___________________________\n\n")
    keyboard.wait('q')
    keyboard.send('ctrl+6')