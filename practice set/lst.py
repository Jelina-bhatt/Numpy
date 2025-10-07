temperature =[30.3, 32.5, 37.5, 38.7]

total=0
for temp in temperature:
    total+=temp

average=total/len(temperature)
print(average)