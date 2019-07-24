from PIL import Image
import os
data=os.listdir()
for e in data:
	try:
		img=Image.open(e)
		img=img.resize((224,224),Image.ANTIALIAS)
		img.save(e)
	except:
		pass