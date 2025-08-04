# Day 3
# Accept a list from the user with 5 elements. Accept name of user and replace the third and fifth elements of the list with the user’s name.
i = 0
l= []
while i < 5:
    element = input(f"Enter element {i + 1} of the list: ")
    l.append(element)
    i += 1
print(l)
name = input("Enter your name: ")
l[2] = name 
l[4] = name
print("Updated list:", l)



# Accept a list with numbers having 10 elements. Remove the third element and all odd elements.
l = []
for i in range(10):
    element = int(input(f"Enter element {i + 1} of the list: "))
    l.append(element)
print("Original:", l)

if len(l) >= 3:
    removed = l.pop(2)  
    print(f"Removed third element: {removed}")

l = [x for x in l if x % 2 == 0]  
print("Updated list after removing 3rd and odd eles:", l)
