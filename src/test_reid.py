from reid import OSNetReID

model = OSNetReID()

embedding = model.extract("data/reference/target.jpg")

print(embedding)
print(embedding.shape)
print(embedding[:10])