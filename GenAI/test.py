l1=[1,45,67,32, 10,109, 1090]
max_n=l1[0]
ordered_list = [l1[0]]

for num in l1:
    if num > max_n:
        max_n = num
        ordered_list.append(num)

print(ordered_list)

