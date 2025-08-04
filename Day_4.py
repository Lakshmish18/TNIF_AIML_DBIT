# Day 4

# Day-4 (4-8-25)
# update existing phone no in the dictionary
phonebook={"Harry":9764543678,"Seema":8975231456,"Ria":7867523145,"Thomas":9825316377}
print(phonebook)

name = input("Enter the name whose number to be modified")
if name in phonebook:
    newnumb = input("Enter the new number")
    phonebook[name] = newnumb
    print("Modified successfully")
    print(phonebook)
else:
    print(name, "was not found.")



# adding new name and phone no in the dictionary
#Add Amir and his nuumber..Key value pair to be added

phonebook={"Harry":8796543908,"Sima":9888765490,"Rita":6779765312}
print(phonebook)
phonebook["Harry"]=9807646889
print(phonebook)
phonebook.update({"Amir":8976543210})
print(phonebook)



# deleting existing phone no in the dictionary
phonebook={"Harry":9764543678,"Seema":8975231456,"Ria":7867523145,"Thomas":9825316377}
print(phonebook)

name = input("enter the name to delete:")
if name in phonebook:
    del phonebook[name]
    print ("delete successfully")
    print (phonebook)
else:
    print (name,"was not found")



# type
phonebook={"Harry":9764543678,"Seema":8975231456,"Ria":7867523145,"Thomas":9825316377}
print(type(phonebook))

#clear
phonebook.clear()
print(phonebook)

print(phonebook.keys())