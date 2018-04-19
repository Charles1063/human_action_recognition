a = '3456789876543456789'
b = '765434567898765456'

#a,b is long int
la = len(a)
lb = len(b)

flag = ''
if la < lb:
	flag = '-'
	a,b = b,a
if la == lb:
	if a == b:
		return '0'
	else:
		i = 0
		for i in xrange (la):
			if a[i] < b[i]:
				flag = '-'
				a,b = b,a
				break
			elif a[i] > b[i]:
				break

# assuming we can do 10 digits operation 
ans = ''
posi = 1
borrow = 0
while posi <= min(la,lb):
	tmp_a,tmp_b = int(a),int(b)
	curr = tmp_a - tmp_b - borrow
	if curr < 0:
		borrow = 1
		ans += str(10 + curr)
	else:
		ans += str(curr)



