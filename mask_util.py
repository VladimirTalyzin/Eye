import os

correct = [ filename for filename in os.listdir("masks-pure") if filename.endswith("png") and not filename.startswith(".") ]

check = { filename: os.path.join("masks-2-pure", filename) for filename in os.listdir("masks-2-pure") if filename.endswith("png") and not filename.startswith(".") }

#for filename, fullname in check.items():
#    if filename in correct:
#        os.rename(fullname, fullname.replace(".png", "!.png"))

for filename, fullname in check.items():
    if filename.endswith("!.png"):
        os.rename(fullname, fullname.replace("!.png", ".png"))