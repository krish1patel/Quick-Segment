from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

lots_of_words = "apple, ocean, bicycle, mountain, stellar, coffee, thunder, whisper, galaxy, bridge, hammer, candle, pocket, velvet, desert, engine, puzzle, silver, window, dragon, forest, jacket, mirror, planet, shadow, ticket, canyon, fabric, island, locket, quartz, rocket, saucer, tunnel, valley, walnut, zenith, anchor, bucket, castle, dinner, flower, garden, harbor, jungle, kettle, ladder, magnet, napkin, orchid, pebble, quiver, ribbon, saddle, temple, umpire, vessel, willow, yellow, zigzag, arctic, banner, cavern, dollar, emblem, fossil, guitar, helmet, indigo, jersey, knight, lantern, meadow, nozzle, orange, pillar, rabbit, shovel, tablet, uproot, vacuum, waffle, xylophone, yachts, zesty, abrupt, bazaar, citrus, domino, editor, falcon, gasket, hiatus, impact, jester, kindle, legend, marble, nebula, oxygen, parade, quartz, radius, safari, timber, unique, vanish, wizard, yachts, zenith, almond, breeze, cobalt, dancer, energy, fridge, glance, hollow, insect, junior, keeper, lizard, mascot, nectar, outlet, picnic, quaint, rhythm, statue, theory, urgent, vortex, window, x-rays, yogurt, zodiac"

# your known classes (do this once at startup, save to disk)
classes = ["person", "car", "dog", "airplane", "bicycle", "cat"]
for word in lots_of_words.split(", "):
    if word not in classes:
        classes.append(word)
class_embeddings = model.encode(classes)  # shape: (6, 384)

# at query time
# query = "give me all the vehicles"
query = input("Enter a word: " )
query_embedding = model.encode([query])  # shape: (1, 384)

scores = cosine_similarity(query_embedding, class_embeddings)[0]
# scores = [0.12, 0.81, 0.09, 0.74, 0.68, 0.08]

# get top matches
top_indices = np.argsort(scores)[::-1][:3]  # top 3
top_classes = [(classes[i], scores[i]) for i in top_indices]
# [("car", 0.81), ("airplane", 0.74), ("bicycle", 0.68)]

print("Most similar words with cosine similarity:")
for index, item in enumerate(top_classes):
    if index >= 5:
        break
    print(item)
