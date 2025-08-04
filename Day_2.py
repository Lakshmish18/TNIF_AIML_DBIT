# Day 2
# program to take a name from user and print all the consonants in the name
def print_consonants(name):
    name = name.lower()  
    vowels = "aeiou"
    consonants = [char for char in name if char.isalpha() and char not in vowels]
    if consonants:
        print("Consonants in the name:", ", ".join(consonants))
name = input("Enter your name: ")
print_consonants(name)

name= input("Enter your name: ")
for i in name:
    if i.lower() not in "aeiou":
        print(i, end=" ")
print("\nConsonants printed successfully.")


# Write a program that keeps accepting strings until ‘Bon’ is entered. The program should display ‘Voyage’ and quit execution

while True:
    n=input("Enter a string (type 'Bon' to quit): ")
    if n.lower() == 'bon':
        print("Voyage")
        break
    else:
        print(f"You entered: {n}. Keep going!")


"""
Write a program that accepts the door number and street of the user and displays only the numbers present in the string. without any libraries used and also no built in fns
Function to extract digits from door number and street name"""

def extract_numbers(address):
    numbers = ""
    index = 0
    while index < len(address):
        ch = address[index]
        if ch >= '0' and ch <= '9':
            numbers = numbers + ch
        index = index + 1
    return numbers

print("Enter your full address:")
address = input("Address: ")
numbers = extract_numbers(address)

if numbers == "":
    print("No numbers were found in your address.")
else:
    print("Extracted numbers from your address:", numbers)



