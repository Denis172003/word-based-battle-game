import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity
from spellchecker import SpellChecker # Import SpellChecker
from scipy.spatial.distance import cosine # Import cosine distance for semantic search
import torch # Import torch for device check

class WordsOfPowerGame:
    def __init__(self):
        """Initialize the Words of Power game with necessary components"""
        # Load the word library and costs
        self.word_library = self._load_word_library()

        # Check for CUDA device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load or initialize the embedding model
        try:
            print("Loading embedding model...")
            # Load model onto the determined device
            self.model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        # Create word embeddings for all counter words
        print("Creating word embeddings...")
        self.counter_embeddings = self._create_counter_embeddings()

        # Create knowledge bases for word relationships
        self.word_relationships = self._build_relationship_database()

        # Define categories and precompute keyword embeddings
        self.categories, self.category_keyword_embeddings = self._define_and_embed_categories()

        # Initialize embedding cache for semantic search
        self.embedding_cache = {}

        # Track game stats
        self.game_history = []

    def _load_word_library(self):
        """Load the word library with costs"""
        word_costs = {
            "Sandpaper": 8, "Oil": 10, "Steam": 15, "Acid": 16, "Gust": 18,
            "Boulder": 20, "Drill": 20, "Vacation": 20, "Fire": 22, "Drought": 24,
            "Water": 25, "Vacuum": 27, "Laser": 28, "Life Raft": 30, "Bear Trap": 32,
            "Hydraulic Jack": 33, "Diamond Cage": 35, "Dam": 35, "Sunshine": 35, "Mutation": 35,
            "Kevlar Vest": 38, "Jackhammer": 38, "Signal Jammer": 40, "Grizzly": 41,
            "Reinforced Steel Door": 42, "Bulldozer": 42, "Sonic Boom": 45, "Robot": 45,
            "Glacier": 45, "Love": 45, "Fire Blanket": 48, "Super Glue": 48, "Therapy": 48,
            "Disease": 50, "Fire Extinguisher": 50, "Satellite": 50, "Confidence": 50,
            "Absorption": 52, "Neutralizing Agent": 55, "Freeze": 55, "Encryption": 55,
            "Proof": 55, "Molotov Cocktail": 58, "Rainstorm": 58, "Viral Meme": 58,
            "War": 59, "Dynamite": 60, "Seismic Dampener": 60, "Propaganda": 60,
            "Explosion": 62, "Lightning": 65, "Evacuation": 65, "Flood": 67,
            "Lava": 68, "Reforestation": 70, "Avalanche": 72, "Earthquake": 74,
            "H-bomb": 75, "Dragon": 75, "Innovation": 75, "Hurricane": 76,
            "Tsunami": 78, "Persistence": 80, "Resilience": 85, "Terraforming Device": 89,
            "Anti-Virus Nanocloud": 90, "AI Kill Switch": 90, "Nanobot Swarm": 92,
            "Reality Resynchronizer": 92, "Cataclysm Containment Field": 92,
            "Solar Deflection Array": 93, "Planetary Evacuation Fleet": 94,
            "Antimatter Cannon": 95, "Planetary Defense Shield": 96, "Singularity Stabilizer": 97,
            "Orbital Laser": 98, "Time": 100
        }

        # Convert to dictionary with IDs
        word_library = {}
        for idx, (word, cost) in enumerate(word_costs.items(), 1):
            word_library[word] = {
                "id": idx,
                "cost": cost
            }

        return word_library

    def _create_counter_embeddings(self):
        """Create embeddings for all counter words"""
        counter_words = list(self.word_library.keys())
        counter_embeddings = {}
        batch_size = 32 # Increased batch size

        print(f"Embedding {len(counter_words)} counter words...")
        # Use the model's encode method directly for efficiency
        embeddings = self.model.encode(counter_words, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        for word, embedding in zip(counter_words, embeddings):
            counter_embeddings[word] = embedding
        print("Counter word embeddings created.")
        return counter_embeddings


    def _build_relationship_database(self):
        """Build an improved database of word relationships with better categorization"""
        relationships = {
            # Food and plants
            "fruit": ["Acid", "Fire", "Drought", "Disease"],
            "vegetable": ["Acid", "Fire", "Drought", "Disease"],
            "banana": ["Acid", "Fire", "Disease"],
            "apple": ["Acid", "Drought", "Disease"],
            "plant": ["Fire", "Drought", "Disease", "Acid"],
            "tree": ["Fire", "Drought", "Bulldozer", "Acid", "Disease"],
            "forest": ["Fire", "Drought", "Bulldozer", "Disease"],
            "garden": ["Drought", "Disease", "Fire"],
            "farm": ["Drought", "Disease", "Fire", "Flood"],

            # Cosmic bodies
            "space": ["Terraforming Device", "Planetary Defense Shield", "Antimatter Cannon"],
            "universe": ["Singularity Stabilizer", "Reality Resynchronizer", "Time"],
            "galaxy": ["Singularity Stabilizer", "Antimatter Cannon", "Reality Resynchronizer"],
            "planet": ["Antimatter Cannon", "H-bomb", "Orbital Laser", "Terraforming Device"], # Added Terraforming Device
            "moon": ["Antimatter Cannon", "Orbital Laser", "Terraforming Device", "Dynamite"], # Added Dynamite for smaller scale
            "asteroid": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon", "Dynamite", "Boulder"], # Added Dynamite, Boulder
            "meteor": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon", "Dynamite", "Boulder"], # Added Dynamite, Boulder
            "comet": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon", "Steam"], # Added Steam for ice
            # Revised Sun/Star counters for more offensive options
            "sun": ["Solar Deflection Array", "Planetary Defense Shield", "Singularity Stabilizer", "Antimatter Cannon", "Reality Resynchronizer", "Time"],
            "star": ["Solar Deflection Array", "Planetary Defense Shield", "Singularity Stabilizer", "Antimatter Cannon", "Reality Resynchronizer", "Time"],
            "black hole": ["Singularity Stabilizer", "Reality Resynchronizer", "Time"], # Added Time
            "radiation": ["Planetary Defense Shield", "Evacuation", "Absorption", "Lead Shielding"], # Lead Shielding not in list, added Absorption
            "snow on satellite dishes": ["Steam"], # Added Steam

            # Small common items
            "plastic bag": ["Fire", "Gust", "Vacuum", "Acid", "Laser"], # Added plastic bag
            "button": ["Drill", "Acid", "Fire", "Laser", "Super Glue"], # Added button
            "zipper": ["Oil", "Drill", "Acid", "Fire", "Super Glue"], # Added zipper
            "screw": ["Drill", "Acid", "Oil", "Jackhammer"], # Added screw
            "bolt": ["Drill", "Acid", "Oil", "Jackhammer"], # Added bolt
            "nail": ["Drill", "Acid", "Jackhammer", "Boulder"], # Added nail
            "needle": ["Kevlar Vest", "Reinforced Steel Door", "Fire", "Laser"], # Added needle
            "pin": ["Kevlar Vest", "Reinforced Steel Door", "Fire", "Laser"], # Added pin
            "paperclip": ["Acid", "Fire", "Drill", "Laser"], # Added paperclip
            "coin": ["Acid", "Fire", "Drill", "Laser"], # Added coin
            "key": ["Acid", "Fire", "Drill", "Laser", "Super Glue"], # Added key


            # Elements and materials
            "fire": ["Water", "Fire Extinguisher", "Fire Blanket", "Vacuum"],
            "water": ["Dam", "Freeze", "Drought", "Absorption"],
            "air": ["Vacuum", "Cataclysm Containment Field"],
            "earth": ["Drill", "Jackhammer", "Bulldozer"],
            "metal": ["Acid", "Laser", "Jackhammer"],
            "wood": ["Fire", "Sandpaper", "Drill", "Acid"],
            "glass": ["Sonic Boom", "Boulder", "Jackhammer"],
            "plastic": ["Fire", "Acid", "Laser"],
            "stone": ["Jackhammer", "Dynamite", "Drill", "Laser"],
            "rock": ["Jackhammer", "Dynamite", "Drill", "Laser"],
            "steel": ["Acid", "Laser", "Jackhammer", "Drill"],
            "gold": ["Acid", "Laser", "Jackhammer"],
            "silver": ["Acid", "Laser", "Jackhammer"],

            # People and human concepts
            "human": ["Disease", "Propaganda", "Viral Meme"],
            "child": ["Love", "Therapy"], # Education not in library
            "children": ["Love", "Therapy"], # Education not in library
            "family": ["Love", "Therapy", "Propaganda"],
            "community": ["Propaganda", "Viral Meme", "Disease"],
            "society": ["Propaganda", "Viral Meme", "War"],
            "crowd": ["Disease", "Propaganda", "Viral Meme"],
            "emotion": ["Therapy", "Love", "Propaganda"],
            "fear": ["Therapy", "Confidence", "Love", "Resilience"],
            "anxiety": ["Therapy", "Confidence", "Love", "Resilience", "Vacation"], # Added Vacation
            "depression": ["Therapy", "Sunshine", "Love", "Resilience", "Vacation"], # Added Vacation
            "love": ["Therapy", "Propaganda", "Time"],
            "hate": ["Love", "Therapy", "Time"],
            "anger": ["Therapy", "Love", "Time"],
            "burnout": ["Vacation"], # Added Vacation
            "mental exhaustion": ["Vacation"], # Added Vacation
            "workaholism": ["Vacation"], # Added Vacation
            "routine boredom": ["Vacation"], # Added Vacation
            "relationship fatigue": ["Vacation"], # Added Vacation
            "study stress": ["Vacation"], # Added Vacation
            "office conflicts": ["Vacation"], # Added Vacation
            "overworking health issues": ["Vacation"], # Added Vacation
            "city noise fatigue": ["Vacation"], # Added Vacation
            "seasonal depression": ["Vacation"], # Added Vacation
            "burnout from innovation jobs": ["Vacation"], # Added Vacation
            "family tensions": ["Vacation"], # Added Vacation
            "student anxiety": ["Vacation"], # Added Vacation
            "pandemic stress": ["Vacation"], # Added Vacation
            "isolation": ["Vacation"], # Added Vacation
            "lack of creativity": ["Vacation"], # Added Vacation
            "overstimulation": ["Vacation"], # Added Vacation
            "corporate monotony": ["Vacation"], # Added Vacation
            "urban claustrophobia": ["Vacation"], # Added Vacation
            "grief": ["Vacation"], # Added Vacation
            "lack of vitamin d": ["Vacation"], # Added Vacation
            "stagnant lifestyles": ["Vacation"], # Added Vacation
            "alienation": ["Vacation"], # Added Vacation
            "nature deprivation": ["Vacation"], # Added Vacation
            "toxic environments": ["Vacation"], # Added Vacation
            "disconnected family ties": ["Vacation"], # Added Vacation
            "hopelessness": ["Vacation"], # Added Vacation
            "fatigue syndromes": ["Vacation"], # Added Vacation
            "monotonous schedules": ["Vacation"], # Added Vacation
            "low self-esteem": ["Vacation"], # Added Vacation
            "hypothermia": ["Steam"], # Added Steam
            "frostbite": ["Steam"], # Added Steam
            "coughing fits": ["Gust"], # Added Gust

            # Natural disasters
            "flood": ["Dam", "Evacuation", "Absorption", "Terraforming Device"],
            "storm": ["Seismic Dampener", "Evacuation", "Cataclysm Containment Field","Thunderstorm"],
            "earthquake": ["Seismic Dampener", "Evacuation", "Terraforming Device"],
            "tornado": ["Cataclysm Containment Field", "Evacuation", "Seismic Dampener"],
            "hurricane": ["Evacuation", "Cataclysm Containment Field", "Planetary Defense Shield"],
            "tsunami": ["Evacuation", "Dam", "Planetary Defense Shield"],
            "drought": ["Rainstorm", "Terraforming Device", "Innovation"],
            "volcano": ["Evacuation", "Seismic Dampener", "Cataclysm Containment Field"],
            "landslide": ["Evacuation", "Seismic Dampener", "Terraforming Device"],
            "avalanche": ["Evacuation", "Seismic Dampener", "Terraforming Device"],
            "ice dams": ["Steam"], # Added Steam
            "ice blockages": ["Steam"], # Added Steam
            "snow-covered roads": ["Steam"], # Added Steam
            "light rain clouds": ["Gust"], # Added Gust

            # Disease and biology
            "disease": ["Anti-Virus Nanocloud", "Evacuation"], # Vaccine, Quarantine not in library
            "virus": ["Anti-Virus Nanocloud"], # Vaccination, Quarantine not in library
            "bacteria": ["Anti-Virus Nanocloud", "Neutralizing Agent", "Acid"], # Added Acid
            "infection": ["Anti-Virus Nanocloud", "Neutralizing Agent"],
            "pandemic": ["Anti-Virus Nanocloud", "Evacuation"], # Quarantine not in library
            "plague": ["Anti-Virus Nanocloud", "Evacuation"], # Quarantine not in library
            "microbe": ["Anti-Virus Nanocloud"], # Vaccine not in library
            "cell": ["Acid", "Disease", "Anti-Virus Nanocloud"],
            "mutation": ["Anti-Virus Nanocloud", "Neutralizing Agent"],
            "bone material": ["Acid"], # Added Acid
            "organic matter decomposition": ["Acid"], # Added Acid
            "bacteria biofilms": ["Acid"], # Added Acid
            "calcareous shells": ["Acid"], # Added Acid
            "eggshells": ["Acid"], # Added Acid
            "mummified remains": ["Acid"], # Added Acid
            "pollen": ["Gust"], # Added Gust

            # Technology and machines
            "technology": ["Water", "Signal Jammer"], # EMP not in library
            "computer": ["Water", "Signal Jammer", "AI Kill Switch"], # EMP not in library
            "laptop": ["Water", "Signal Jammer"], # EMP not in library
            "phone": ["Water", "Signal Jammer"], # EMP not in library
            "robot": ["Water", "Signal Jammer", "AI Kill Switch"], # EMP not in library
            "electronic": ["Water", "Signal Jammer"], # EMP not in library
            "machine": ["Water", "Acid"], # EMP not in library
            "engine": ["Water", "Acid", "Oil", "Steam"], # EMP not in library, Added Oil, Steam
            "motor": ["Water", "Acid", "Oil"], # EMP not in library, Added Oil
            "device": ["Water", "Signal Jammer"], # EMP not in library
            "dry machine gears": ["Oil"], # Added Oil
            "squeaky hinges": ["Oil"], # Added Oil
            "stuck bolts": ["Oil"], # Added Oil
            "frozen locks": ["Oil", "Steam"], # Added Oil, Steam
            "friction in pistons": ["Oil"], # Added Oil
            "saws getting stuck": ["Oil"], # Added Oil
            "bicycle chain squeaking": ["Oil"], # Added Oil
            "dull mechanical parts": ["Oil"], # Added Oil
            "drill bit overheating": ["Oil"], # Added Oil
            "engine grinding": ["Oil"], # Added Oil
            "worn tool joints": ["Oil"], # Added Oil
            "stiff robotic arms": ["Oil"], # Added Oil
            "noisy door joints": ["Oil"], # Added Oil
            "dry conveyor belts": ["Oil"], # Added Oil
            "dull razor blades": ["Oil"], # Added Oil
            "chainsaw chains": ["Oil"], # Added Oil
            "dry o-rings": ["Oil"], # Added Oil
            "jammed typewriters": ["Oil"], # Added Oil
            "worn sewing machine gears": ["Oil"], # Added Oil
            "old fan motors": ["Oil"], # Added Oil
            "dry cable wires": ["Oil"], # Added Oil
            "brake part seizing": ["Oil"], # Added Oil
            "pulley friction": ["Oil"], # Added Oil
            "metal cutting friction": ["Oil"], # Added Oil
            "stuck valve stems": ["Oil"], # Added Oil
            "dry bike brakes": ["Oil"], # Added Oil
            "dry automotive gaskets": ["Oil"], # Added Oil
            "jammed sliding doors": ["Oil"], # Added Oil
            "squeaking treadmill belts": ["Oil"], # Added Oil
            "frozen gears": ["Steam"], # Added Steam
            "frozen fuel lines": ["Steam"], # Added Steam
            "cold machinery startup": ["Steam"], # Added Steam
            "chilled medical tools": ["Steam"], # Added Steam
            "frozen water filters": ["Steam"], # Added Steam
            "frozen ventilation systems": ["Steam"], # Added Steam
            "frozen brake cables": ["Steam"], # Added Steam
            "frozen ropes": ["Steam"], # Added Steam
            "lightweight drones": ["Gust"], # Added Gust
            "smoke alarms": ["Gust"], # Added Gust
            "lock mechanisms": ["Drill"], # Added Drill
            "weak safes": ["Drill"], # Added Drill
            "batteries with corrosion": ["Acid"], # Added Acid
            "electrical poles": ["Boulder"], # Added Boulder
            "solar panels on ground": ["Boulder"], # Added Boulder
            "sewer covers": ["Drill"], # Added Drill
            "overheating machines": ["Water"], # Added Water
            "overcooked electronics": ["Water"], # Added Water
            "optical sensors": ["Laser"], # Added Laser
            "solar panels": ["Laser"], # Added Laser
            "drones": ["Laser"], # Added Laser
            "cameras": ["Laser"], # Added Laser
            "infrared devices": ["Laser"], # Added Laser
            "smoke detectors": ["Laser"], # Added Laser
            "light-based communication systems": ["Laser"], # Added Laser
            "satellites": ["Laser"], # Added Laser
            "security cameras": ["Laser"], # Added Laser
            "military infrared tracking": ["Laser"], # Added Laser
            "robot vision systems": ["Laser"], # Added Laser
            "solar-powered gadgets": ["Laser"], # Added Laser
            "power grids": ["Laser"], # Added Laser
            "light-sensitive materials": ["Laser"], # Added Laser
            "laser-guided weapons": ["Laser"], # Added Laser
            "optical fiber networks": ["Laser"], # Added Laser
            "laser cut jobs": ["Laser"], # Added Laser
            "night-vision equipment": ["Laser"], # Added Laser
            "thermal detectors": ["Laser"], # Added Laser
            "fire alarms": ["Laser"], # Added Laser
            "modern televisions": ["Laser"], # Added Laser
            "laser-based communication": ["Laser"], # Added Laser
            "targeted surveillance systems": ["Laser"], # Added Laser
            "photocells and photoelectric devices": ["Laser"], # Added Laser
            "sensors in medical devices": ["Laser"], # Added Laser
            "airborne targets": ["Laser"], # Added Laser
            "optical drive systems": ["Laser"], # Added Laser
            "printed barcodes": ["Laser"], # Added Laser

            "vehicle": ["Bear Trap", "Oil"], # EMP not in library
            "car": ["Bear Trap", "Oil", "Boulder"], # EMP not in library, Added Oil, Boulder
            "truck": ["Bear Trap", "Oil"], # EMP not in library, Added Oil
            "bus": ["Bear Trap", "Oil", "Fire"], # EMP not in library, Added Oil, Fire
            "train": ["Earthquake", "Oil"], # EMP not in library, Added Oil
            "aircraft": ["Signal Jammer", "Dragon", "Steam", "Laser", "Orbital Laser", "Hurricane", "Explosion", "War"], # Added specific aircraft counters
            "helicopter": ["Signal Jammer", "Dragon", "Laser", "Orbital Laser", "Hurricane", "Explosion", "War", "Gust"], # Added specific helicopter counters, Gust for small ones
            "airplane": ["Signal Jammer", "Dragon", "Steam", "Laser", "Orbital Laser", "Hurricane", "Explosion", "War", "Fire"], # Added specific airplane counters
            "drone": ["Signal Jammer", "Laser", "Gust", "Water", "AI Kill Switch", "Orbital Laser"], # Added specific drone counters
            "ship": ["Hurricane", "Tsunami", "Acid", "Drill"], # Added Drill
            "boat": ["Hurricane", "Tsunami", "Acid", "Fire", "Steam"], # Added Fire, Steam
            "submarine": ["Signal Jammer", "Acid"], # Depth Charges not in library
            "tank": ["H-bomb", "Orbital Laser", "Dynamite"],
            "car frames": ["Drill"], # Added Drill
            "ship hulls": ["Drill"], # Added Drill
            "frost on car windows": ["Steam"], # Added Steam
            "ice on planes": ["Steam"], # Added Steam
            "frozen boats": ["Steam"], # Added Steam
            "hot air balloons": ["Gust"], # Added Gust
            "small gliders": ["Gust"], # Added Gust
            "papercraft planes": ["Gust"], # Added Gust
            "open umbrellas": ["Gust"], # Added Gust
            "rvs": ["Fire"], # Added Fire
            # "airplanes": ["Fire"], # Redundant, covered by "airplane" above
            "rust on bicycles": ["Sandpaper"], # Added Sandpaper
            "tar on vehicles": ["Acid"], # Added Acid
            "gasoline engines": ["Fire"], # Added Fire
            "rubber tires": ["Fire"], # Added Fire

            # Structures and buildings
            "building": ["Earthquake", "Dynamite", "Bulldozer", "H-bomb", "Lava", "Boulder"], # Added Boulder
            "house": ["Fire", "Earthquake", "Bulldozer", "Boulder"], # Added Boulder, Fire
            "skyscraper": ["Earthquake", "Dynamite", "H-bomb"],
            "tower": ["Earthquake", "Dynamite", "H-bomb"],
            "bridge": ["Earthquake", "Dynamite", "Flood", "Boulder"], # Added Boulder
            "dam": ["Earthquake", "Dynamite", "Flood", "Boulder"], # Added Boulder
            "wall": ["Bulldozer", "Dynamite", "Jackhammer", "Laser", "Drill", "Boulder"],
            "pillar": ["Drill", "Jackhammer", "Dynamite", "Bulldozer", "Earthquake"], # Added pillar relationship# Added Drill, Boulder
            "barrier": ["Bulldozer", "Dynamite", "Jackhammer", "Laser", "Acid"], # Added Acid
            "fence": ["Bulldozer", "Dynamite", "Laser", "Boulder"], # Added Boulder
            "door": ["Bulldozer", "Dynamite", "Laser", "Drill"], # Added Drill
            "concrete slabs": ["Drill"], # Added Drill
            "wooden planks": ["Drill"], # Added Drill
            "tile floors": ["Drill"], # Added Drill
            "plastic walls": ["Drill"], # Added Drill
            "asphalt roads": ["Drill"], # Added Drill
            "ice dams on roofs": ["Steam"], # Added Steam
            "icy sidewalks": ["Steam"], # Added Steam
            "freezer stuck doors": ["Steam"], # Added Steam
            "icy handrails": ["Steam"], # Added Steam
            "icy doorsteps": ["Steam"], # Added Steam
            "wooden fences": ["Boulder"], # Added Boulder
            "mud huts": ["Boulder"], # Added Boulder
            "stone walls": ["Boulder"], # Added Boulder
            "castle gates": ["Boulder"], # Added Boulder
            "fortified trenches": ["Boulder"], # Added Boulder
            "palisades": ["Boulder"], # Added Boulder
            "wooden watchtowers": ["Boulder"], # Added Boulder
            "hunting blinds": ["Boulder"], # Added Boulder
            "sandbag walls": ["Boulder"], # Added Boulder
            "shacks": ["Boulder"], # Added Boulder
            "market stalls": ["Boulder"], # Added Boulder
            "temporary stages": ["Boulder"], # Added Boulder
            "billboard poles": ["Boulder"], # Added Boulder
            "old bridges": ["Boulder"], # Added Boulder
            "flammable houses": ["Fire"], # Added Fire
            "straw roofs": ["Fire"], # Added Fire
            "old warehouses": ["Fire"], # Added Fire
            "bookstores": ["Fire"], # Added Fire
            "churches": ["Fire"], # Added Fire
            "old city blocks": ["Fire"], # Added Fire
            "campsites": ["Fire"], # Added Fire
            "fuel stations": ["Fire"], # Added Fire
            "energy plants": ["Fire"], # Added Fire
            "timber industries": ["Fire"], # Added Fire
            "old window frames": ["Sandpaper"], # Added Sandpaper
            "uneven drywall patches": ["Sandpaper"], # Added Sandpaper
            "concrete splatter": ["Sandpaper"], # Added Sandpaper
            "old grout": ["Acid"], # Added Acid
            "cement": ["Acid"], # Added Acid
            "soap scum buildup": ["Acid"], # Added Acid
            "mold stains": ["Acid"], # Added Acid
            "paint layers": ["Acid"], # Added Acid
            "graffiti on walls": ["Acid"], # Added Acid
            "hair clogs": ["Acid"], # Added Acid
            "wood structures": ["Fire"], # Added Fire
            "straw huts": ["Fire"], # Added Fire
            "timber fences": ["Fire"], # Added Fire
            "gas pipelines": ["Fire"], # Added Fire
            "electrical wires": ["Fire"], # Added Fire
            "hay bales": ["Fire"], # Added Fire
            "paper trash piles": ["Fire"], # Added Fire
            "cardboard boxes": ["Fire"], # Added Fire
            "books": ["Fire"], # Added Fire
            "reservoirs": ["Drought"], # Added Drought
            "water fountains in parks": ["Drought"], # Added Drought
            "water treatment plants": ["Drought"], # Added Drought
            "drinking water sources": ["Drought"], # Added Drought
            "dust": ["Vacuum"], # Added Vacuum
            "spilled liquids": ["Vacuum"], # Added Vacuum
            "airborne bacteria": ["Vacuum"], # Added Vacuum
            "polluted air": ["Vacuum"], # Added Vacuum
            "small debris": ["Vacuum"], # Added Vacuum
            "carpet fibers": ["Vacuum"], # Added Vacuum
            "dirt on hard floors": ["Vacuum"], # Added Vacuum
            "pet hair": ["Vacuum"], # Added Vacuum
            "ash": ["Vacuum"], # Added Vacuum
            "stale air": ["Vacuum"], # Added Vacuum
            "clothing lint": ["Vacuum"], # Added Vacuum
            "old cobwebs": ["Vacuum"], # Added Vacuum
            "household odors": ["Vacuum"], # Added Vacuum
            "mold spores": ["Vacuum"], # Added Vacuum
            "lightweight plastics": ["Vacuum"], # Added Vacuum
            "small insects": ["Vacuum"], # Added Vacuum
            "broken glass shards": ["Vacuum"], # Added Vacuum
            "leaf litter": ["Vacuum"], # Added Vacuum
            "food crumbs": ["Vacuum"], # Added Vacuum
            "dust mites": ["Vacuum"], # Added Vacuum
            "pollen": ["Vacuum"], # Added Vacuum
            "smoke particles": ["Vacuum"], # Added Vacuum
            "microscopic dust": ["Vacuum"], # Added Vacuum
            "dry leaves": ["Vacuum"], # Added Vacuum
            "soot particles": ["Vacuum"], # Added Vacuum
            "carpet stains": ["Vacuum"], # Added Vacuum
            "static charge": ["Vacuum"], # Added Vacuum
            "small food scraps": ["Vacuum"], # Added Vacuum

            # Weapons and military
            "weapon": ["Kevlar Vest", "Reinforced Steel Door", "Diamond Cage"],
            "gun": ["Kevlar Vest", "Reinforced Steel Door"],
            "bullet": ["Kevlar Vest", "Reinforced Steel Door"],
            "knife": ["Kevlar Vest", "Reinforced Steel Door"],
            "bomb": ["Kevlar Vest", "Reinforced Steel Door", "Diamond Cage"],
            "missile": ["Kevlar Vest", "Reinforced Steel Door", "Diamond Cage"],
            "explosive": ["Kevlar Vest", "Reinforced Steel Door", "Diamond Cage"],
            "army": ["H-bomb", "Dragon", "War"],
            "navy": ["Hurricane", "Tsunami", "Orbital Laser"],
            "air force": ["Dragon", "Orbital Laser"], # Missile Defense System not in library
            "military": ["H-bomb", "Dragon", "War"],
            "soldier": ["Disease", "Propaganda", "War"],
            "fighter": ["Dragon", "Orbital Laser"], # Missile Defense System not in library
            "armor plates": ["Drill"], # Added Drill
            "smoke bombs": ["Gust"], # Added Gust
            "smoke grenades": ["Gust"], # Added Gust
            "army barricades": ["Boulder"], # Added Boulder
            "barbed wire": ["Boulder"], # Added Boulder
            "fireworks storage": ["Fire"], # Added Fire
            "fireworks": ["Fire"], # Added Fire
            "gasoline tanks": ["Fire"], # Added Fire
            "chemical containers": ["Fire"], # Added Fire
            "flammable chemicals": ["Fire"], # Added Fire
            "explosives": ["Water"], # Added Water

            # Information and communication
            "information": ["Encryption", "Signal Jammer", "Viral Meme"],
            "data": ["Encryption", "Signal Jammer"],
            "signal": ["Signal Jammer", "Encryption"],
            "communication": ["Signal Jammer", "Encryption"],
            "message": ["Encryption", "Signal Jammer", "Viral Meme"],
            "news": ["Propaganda", "Viral Meme", "Signal Jammer"],
            "media": ["Propaganda", "Viral Meme", "Signal Jammer"],
            "social media": ["Viral Meme", "Signal Jammer", "Propaganda"],
            "internet": ["Signal Jammer", "Encryption", "Viral Meme"],
            "network": ["Signal Jammer", "Encryption", "Viral Meme"],
            "paper documents": ["Fire"], # Added Fire

            # Weather and climate
            "weather": ["Terraforming Device", "Cataclysm Containment Field"],
            "climate": ["Terraforming Device", "Cataclysm Containment Field"],
            "heat": ["Water", "Freeze"], # Ice not in library
            "cold": ["Fire", "Sunshine", "Lava", "Steam"], # Added Steam
            "rain": ["Drought", "Absorption"],
            "snow": ["Sunshine", "Fire", "Lava"],
            "wind": ["Cataclysm Containment Field", "Planetary Defense Shield"],
            "fog": ["Sunshine", "Laser", "Gust"], # Wind not in library (Gust is), Added Gust
            "cloud": ["Sunshine", "Laser", "Gust"], # Wind not in library (Gust is), Added Gust
            "lightning": ["Reinforced Steel Door", "Absorption"], # Grounding Rod not in library
            "smoke clouds": ["Gust"], # Added Gust
            "dust clouds": ["Gust"], # Added Gust
            "falling ash": ["Gust"], # Added Gust
            "stagnant air": ["Gust"], # Added Gust
            "smog": ["Gust"], # Added Gust
            "freezing temperatures": ["Steam"], # Added Steam
            "wildfires": ["Fire"], # Added Fire
            "heatstroke": ["Water"], # Added Water
            "thirst": ["Water"], # Added Water
            "desertification": ["Water", "Drought"], # Added Water, Drought
            "fire hazards": ["Water"], # Added Water
            "dust storms": ["Water"], # Added Water
            "airborne pollutants": ["Water"], # Added Water
            "extreme temperatures": ["Water"], # Added Water
            "high heat environments": ["Water"], # Added Water
            "accumulated smoke particles": ["Water"], # Added Water
            "earthquake dust": ["Water"], # Added Water

            # Abstract concepts
            "idea": ["Propaganda", "Viral Meme", "Innovation"],
            "concept": ["Propaganda", "Viral Meme", "Innovation"],
            "theory": ["Proof", "Innovation", "Propaganda"],
            "belief": ["Propaganda", "Proof", "Viral Meme"],
            "movement": ["Propaganda", "War", "Viral Meme"],
            "innovation": ["Time", "Innovation", "Persistence"],
            "progress": ["Time", "Innovation", "Persistence"],
            "change": ["Persistence", "Innovation", "Time"],
            "future": ["Time", "Innovation", "Reality Resynchronizer"],
            "past": ["Time", "Reality Resynchronizer", "Innovation"],
            "time": ["Time", "Reality Resynchronizer", "Singularity Stabilizer"],
            "bad odors": ["Gust"], # Added Gust
            "dehydration": ["Water"], # Added Water
            "tiredness": ["Water"], # Added Water

            # Space and cosmic objects
            "space": ["Terraforming Device", "Planetary Defense Shield", "Antimatter Cannon"],
            "universe": ["Singularity Stabilizer", "Reality Resynchronizer", "Time"],
            "galaxy": ["Singularity Stabilizer", "Antimatter Cannon", "Reality Resynchronizer"],
            "planet": ["Antimatter Cannon", "H-bomb", "Orbital Laser"],
            "moon": ["Antimatter Cannon", "Orbital Laser", "Terraforming Device"],
            "asteroid": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon"],
            "meteor": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon"],
            "comet": ["Planetary Defense Shield", "Orbital Laser", "Antimatter Cannon"],
            "sun": ["Solar Deflection Array", "Planetary Defense Shield"],
            "star": ["Solar Deflection Array", "Planetary Defense Shield"],
            "black hole": ["Singularity Stabilizer", "Reality Resynchronizer"],
            "radiation": ["Planetary Defense Shield", "Evacuation"], # Lead Shielding not in library
            "snow on satellite dishes": ["Steam"], # Added Steam

            # Supernatural and mythological
            "monster": ["Dragon", "H-bomb", "Diamond Cage"],
            "ghost": ["Vacuum", "Reality Resynchronizer"],
            "spirit": ["Reality Resynchronizer", "Vacuum", "Time"],
            "demon": ["Reality Resynchronizer", "Dragon", "H-bomb"],
            "angel": ["Reality Resynchronizer", "Dragon", "Time"],
            "god": ["Reality Resynchronizer", "Time", "Singularity Stabilizer"],
            "vampire": ["Sunshine", "Fire", "Dragon"],
            "werewolf": ["Fire", "Dragon"], # Silver Bullet not in library
            "zombie": ["Fire", "H-bomb", "Orbital Laser"],
            "dragon": ["H-bomb", "Orbital Laser", "Singularity Stabilizer"],
            "magic": ["Reality Resynchronizer"], # Science, Technology not in library
            "spell": ["Reality Resynchronizer"], # Science, Technology not in library
            "giant spider webs": ["Boulder"], # Added Boulder

            # Protection and defense
            "shield": ["Acid", "Laser", "Dynamite", "H-bomb"],
            "armor": ["Acid", "Laser", "Dynamite", "H-bomb"],
            "defense": ["Acid", "Laser", "Dynamite", "H-bomb"],
            "protection": ["Acid", "Laser", "Dynamite", "H-bomb"],
            "security": ["Signal Jammer"], # Virus, Hacker not in library
            "guard": ["Disease", "Propaganda", "War"],
            "lock": ["Acid", "Laser", "Dynamite"],
            "encryption": ["Signal Jammer"], # Quantum Computer, AI not in library
            "roadblocks": ["Boulder"], # Added Boulder

            # Power and energy
            "energy": ["Absorption", "Vacuum", "Singularity Stabilizer"],
            "power": ["Absorption", "Vacuum", "Singularity Stabilizer"],
            "electricity": ["Water"], # EMP, Short Circuit not in library
            "nuclear": ["Evacuation"], # Containment Field, Lead Shielding not in library
            "battery": ["Water", "Acid"], # Short Circuit not in library
            "generator": ["Water", "Acid"], # EMP not in library
            "solar": [], # Cloud Cover, Dust Storm, Night not in library
            "wind power": ["Earthquake"], # Still Air, EMP not in library
            "frozen pipes": ["Steam"], # Added Steam
            "candle flames": ["Gust"], # Added Gust
            "flying embers": ["Gust"], # Added Gust
            "oil spills": ["Fire"], # Added Fire
            "gasoline": ["Fire"], # Added Fire

            # Generic threat categories (fallbacks)
            "threat": ["Planetary Defense Shield", "Resilience", "H-bomb", "Singularity Stabilizer"],
            "danger": ["Evacuation", "Resilience", "Planetary Defense Shield"],
            "attack": ["Kevlar Vest", "Reinforced Steel Door", "Planetary Defense Shield"],
            "catastrophe": ["Evacuation", "Cataclysm Containment Field", "Planetary Defense Shield"],
            "unknown": ["Time", "Innovation", "Resilience", "Planetary Defense Shield", "Singularity Stabilizer"],
            "mystery": ["Time", "Innovation", "Reality Resynchronizer"],
            "chaos": ["Reality Resynchronizer", "Singularity Stabilizer"], # Order not in library
            "paradox": ["Reality Resynchronizer", "Time", "Singularity Stabilizer"],

             # Added from user lists:
            # Sandpaper
            "rough surfaces": ["Sandpaper"],
            "rusty metal": ["Sandpaper"],
            "peeling paint": ["Sandpaper"],
            "uneven wood": ["Sandpaper"],
            "splinters": ["Sandpaper"],
            "dried glue spots": ["Sandpaper"],
            "sharp wood edges": ["Sandpaper"],
            "tarnished silver": ["Sandpaper"],
            "dull knife handles": ["Sandpaper"],
            "dirty tool handles": ["Sandpaper"],
            "corroded pipes": ["Sandpaper"],
            "surface bumps": ["Sandpaper"],
            "fiberglass imperfections": ["Sandpaper"],
            "plastic rough patches": ["Sandpaper"],
            "hardened mud on surfaces": ["Sandpaper"],
            "wood rot": ["Sandpaper"],
            "sticky residues": ["Sandpaper"],
            "dried paint blobs": ["Sandpaper"],
            "sharp corners": ["Sandpaper"],
            "ice on wood": ["Sandpaper"],
            "small nails sticking out": ["Sandpaper"],
            "moss on wood decks": ["Sandpaper"],
            "rough skateboard grip": ["Sandpaper"],
            "car scratch touch-ups": ["Sandpaper"],
            "worn-out furniture finish": ["Sandpaper"],
            "unfinished wooden sculptures": ["Sandpaper"],
            # Oil
            "rust buildup": ["Oil"],
            # Steam
            "cold hospital rooms": ["Steam"],
            # Acid
            "rust": ["Acid"],
            "stubborn mineral stains": ["Acid"],
            "lime deposits": ["Acid"],
            "algae on surfaces": ["Acid"],
            "food stuck to pans": ["Acid"],
            "metal oxidation": ["Acid"],
            "dirt encrustations": ["Acid"],
            "old varnish": ["Acid"],
            "tree sap": ["Acid"],
            "ceramic deposits": ["Acid"],
            "ink stains": ["Acid"],
            "hardened grease": ["Acid"],
            "tough biological stains": ["Acid"],
            "dirty jewelry": ["Acid"],
            # Gust
            "balloons": ["Gust"],
            "flying leaves": ["Gust"],
            "dandelions spreading seeds": ["Gust"],
            "light pollution": ["Gust"], # (if carrying particles)
            "spiders on webs": ["Gust"],
            "falling lightweight construction material": ["Gust"],
            # Boulder
            "tents": ["Boulder"],
            "campfires": ["Boulder"], # (snuffing)
            # Drill
            "rock walls": ["Drill"],
            "thick ice blocks": ["Drill"],
            "frozen ground": ["Drill"],
            "stone sculptures": ["Drill"], # (carving)
            "metal sheets": ["Drill"],
            "glass": ["Drill"], # (careful)
            "bones": ["Drill"], # (surgery)
            "coral reefs": ["Drill"], # (scientific)
            "mountainsides": ["Drill"], # (tunneling)
            "mine shafts": ["Drill"],
            "sandstone layers": ["Drill"],
            "fossil beds": ["Drill"],
            "tree trunks": ["Drill"],
            "wooden doors": ["Drill"],
            "ceramic pots": ["Drill"],
            "soft metal layers": ["Drill"],
            "ice fishing holes": ["Drill"],
            "mudstone": ["Drill"],
            # Fire
            "dry bushes": ["Fire"],
            "paper documents": ["Fire"],
            "grass fields": ["Fire"],
            "trash piles": ["Fire"],
            "dry leaves": ["Fire"],
            "airborne particles": ["Fire"], # (explosive)
            "clothing": ["Fire"],
            "curtains": ["Fire"],
            "crop fields": ["Fire"],
            # Drought
            "rice paddies": ["Drought"],
            "corn fields": ["Drought"],
            "wheat fields": ["Drought"],
            "grape vines": ["Drought"],
            "citrus orchards": ["Drought"],
            "avocado farms": ["Drought"],
            "pine forests": ["Drought"],
            "oak forests": ["Drought"],
            "bamboo groves": ["Drought"],
            "grasslands": ["Drought"],
            "savannahs": ["Drought"],
            "rainforests": ["Drought"],
            "coffee plantations": ["Drought"],
            "palm plantations": ["Drought"],
            "small lakes": ["Drought"],
            "wetlands": ["Drought"],
            "mangrove forests": ["Drought"],
            "fish ponds": ["Drought"],
            "duck farms": ["Drought"],
            "dairy farms": ["Drought"], # (indirectly)
            "water supply tanks": ["Drought"],
            "city water systems": ["Drought"],
            "golf courses": ["Drought"],
            "parks": ["Drought"],
            "public fountains": ["Drought"],
            "swimming pools": ["Drought"],
            "hydroelectric dams": ["Drought"],
            "ponds": ["Drought"],
            "rivers": ["Drought"],
            "streams": ["Drought"],
            "frozen food": ["Steam"], # Added Steam
            "dry grass": ["Fire"], # Added Fire
            "petrol": ["Fire"], # Added Fire
            "cotton clothing": ["Fire"], # Added Fire
            "dry brush": ["Fire"], # Added Fire
            "old forests": ["Fire"], # Added Fire
            "furniture": ["Fire"], # Added Fire
            "dry crops": ["Fire"], # Added Fire
            "camping gear": ["Fire"], # Added Fire
            "cotton fields": ["Fire"], # Added Fire
            "wheat farms": ["Drought"], # Added Drought
            "fruit orchards": ["Drought"], # Added Drought
            "vineyards": ["Drought"], # Added Drought
            "pastures for grazing": ["Drought"], # Added Drought
            "mangroves": ["Drought"], # Added Drought
            "coffee farms": ["Drought"], # Added Drought
            "cocoa plantations": ["Drought"], # Added Drought
            "palm groves": ["Drought"], # Added Drought
            "avocado orchards": ["Drought"], # Added Drought
            "irrigated farmlands": ["Drought"], # Added Drought
            "water-hungry industries": ["Drought"], # Added Drought
            "cattle ranches": ["Drought"], # Added Drought
            "hydroponic farming": ["Drought"], # Added Drought
            "floriculture": ["Drought"], # Added Drought
            "aquaculture": ["Drought"], # Added Drought
            "ecosystems dependent on fresh water": ["Drought"], # Added Drought
            "overcooked food": ["Water"], # Added Water
            "oil slicks": ["Water"], # Added Water
            "dry skin": ["Water"], # Added Water
            "dry eyes": ["Water"], # Added Water
            "hot beverages": ["Water"], # Added Water
            "chemical reactions": ["Water"], # Added Water
            "dry hair": ["Water"], # Added Water
            "stuck objects": ["Water"], # Added Water
            "burns": ["Water"], # Added Water
            "skin wounds": ["Water"], # Added Water
        }

        # Helper function to add counters safely
        def add_counter(keyword, counter_word):
            keyword_lower = keyword.lower().strip()
            if not keyword_lower: return # Skip empty keywords
            if keyword_lower not in relationships:
                relationships[keyword_lower] = []
            if counter_word not in relationships[keyword_lower]:
                relationships[keyword_lower].append(counter_word)

        # Add Sandpaper counters
        sandpaper_targets = ["Rough surfaces", "Rusty metal", "Peeling paint", "Uneven wood", "Splinters", "Dried glue spots", "Sharp wood edges", "Tarnished silver", "Dull knife handles", "Dirty tool handles", "Corroded pipes", "Surface bumps", "Fiberglass imperfections", "Plastic rough patches", "Hardened mud on surfaces", "Wood rot", "Sticky residues", "Dried paint blobs", "Sharp corners", "Ice on wood", "Small nails sticking out", "Moss on wood decks", "Rough skateboard grip", "Car scratch touch-ups", "Uneven drywall patches", "Worn-out furniture finish", "Concrete splatter", "Old window frames", "Unfinished wooden sculptures", "Rust on bicycles"]
        for target in sandpaper_targets: add_counter(target, "Sandpaper")

        # Add Oil counters
        oil_targets = ["Dry machine gears", "Rust buildup", "Squeaky hinges", "Stuck bolts", "Frozen locks", "Friction in pistons", "Saws getting stuck", "Bicycle chain squeaking", "Dull mechanical parts", "Drill bit overheating", "Engine grinding", "Worn tool joints", "Stiff robotic arms", "Noisy door joints", "Dry conveyor belts", "Dull razor blades", "Chainsaw chains", "Dry O-rings", "Jammed typewriters", "Worn sewing machine gears", "Old fan motors", "Dry cable wires", "Brake part seizing", "Pulley friction", "Metal cutting friction", "Stuck valve stems", "Dry bike brakes", "Dry automotive gaskets", "Jammed sliding doors", "Squeaking treadmill belts"]
        for target in oil_targets: add_counter(target, "Oil")

        # Add Steam counters
        steam_targets = ["Frozen pipes", "Frost on car windows", "Icy locks", "Frozen gears", "Hypothermia", "Cold hospital rooms", "Freezing temperatures", "Frozen fuel lines", "Ice on planes", "Frostbite treatment", "Ice dams on roofs", "Icy sidewalks", "Freezer stuck doors", "Cold machinery startup", "Ice blockages", "Chilled medical tools", "Frozen water filters", "Ice sculptures", "Snow-covered roads", "Frozen boats", "Ice inside engines", "Frozen ventilation systems", "Snow on satellite dishes", "Icy handrails", "Stuck zippers", "Frozen brake cables", "Icy doorsteps", "Frozen food", "Frost inside freezers", "Frozen ropes"]
        for target in steam_targets: add_counter(target, "Steam")

        # Add Acid counters
        acid_targets = ["Rust", "Stubborn mineral stains", "Lime deposits", "Old grout", "Algae on surfaces", "Cement", "Batteries with corrosion", "Soap scum buildup", "Food stuck to pans", "Metal oxidation", "Bone material", "Dirt encrustations", "Mold stains", "Old varnish", "Tree sap", "Paint layers", "Ceramic deposits", "Tar on vehicles", "Ink stains", "Graffiti on walls", "Organic matter decomposition", "Hardened grease", "Hair clogs", "Tough biological stains", "Bacteria biofilms", "Calcareous shells", "Eggshells", "Concrete barriers", "Mummified remains", "Dirty jewelry"]
        for target in acid_targets: add_counter(target, "Acid")

        # Add Gust counters
        gust_targets = ["Smoke clouds", "Fog", "Bad odors", "Dust clouds", "Lightweight drones", "Balloons", "Insects", "Falling ash", "Pollen in the air", "Campfire smoke", "Smoke alarms", "Light pollution", "Spiders on webs", "Flying leaves", "Small birds in mid-flight", "Hot air balloons", "Candle flames", "Flying embers", "Dandelions spreading seeds", "Mosquito clouds", "Stagnant air", "Smog", "Small gliders", "Papercraft planes", "Open umbrellas", "Smoke bombs", "Smoke grenades", "Coughing fits", "Falling lightweight construction material", "Light rain clouds"]
        for target in gust_targets: add_counter(target, "Gust")

        # Add Boulder counters
        boulder_targets = ["Small trees", "Cars", "Wooden fences", "House walls", "Weak bridges", "Light buildings", "Tents", "Roadblocks", "Small animals", "Human-made dams", "Stone walls", "Mud huts", "Electrical poles", "Barbed wire", "Army barricades", "Giant spider webs", "Castle gates", "Fortified trenches", "Palisades", "Wooden watchtowers", "Hunting blinds", "Sandbag walls", "Shacks", "Fences", "Campfires", "Market stalls", "Temporary stages", "Billboard poles", "Solar panels on ground", "Old bridges"]
        for target in boulder_targets: add_counter(target, "Boulder")

        # Add Drill counters
        drill_targets = ["Rock walls", "Concrete slabs", "Wooden planks", "Thick ice blocks", "Frozen ground", "Stone sculptures", "Metal sheets", "Walls", "Armor plates", "Car frames", "Weak safes", "Lock mechanisms", "Glass", "Bones", "Coral reefs", "Mountainsides", "Mine shafts", "Sandstone layers", "Fossil beds", "Tree trunks", "Wooden doors", "Tile floors", "Plastic walls", "Ceramic pots", "Soft metal layers", "Sewer covers", "Asphalt roads", "Ice fishing holes", "Ship hulls", "Mudstone"]
        for target in drill_targets: add_counter(target, "Drill")

        # Add Vacation counters
        vacation_targets = ["Burnout", "Mental exhaustion", "Workaholism", "Routine boredom", "Relationship fatigue", "Study stress", "Office conflicts", "Overworking health issues", "City noise fatigue", "Seasonal depression", "Burnout from innovation jobs", "Family tensions", "Student anxiety", "Pandemic stress", "Isolation", "Lack of creativity", "Overstimulation", "Corporate monotony", "Urban claustrophobia", "Grief", "Lack of vitamin D", "Stagnant lifestyles", "Alienation", "Nature deprivation", "Toxic environments", "Disconnected family ties", "Hopelessness", "Fatigue syndromes", "Monotonous schedules", "Low self-esteem"]
        for target in vacation_targets: add_counter(target, "Vacation")

        # Add Fire counters (List 11)
        fire_targets_11 = ["Wood structures", "Dry grass", "Paper documents", "Leaves", "Oil spills", "Plastic materials", "Petrol", "Cotton clothing", "Tents", "Straw huts", "Dry brush", "Old forests", "Wildfires", "Buses", "Furniture", "Dry crops", "Timber fences", "Fireworks", "Gasoline tanks", "Gas pipelines", "Chemical containers", "Electrical wires", "Camping gear", "Hay bales", "Flammable chemicals", "Rubber tires", "Paper trash piles", "Cotton fields", "Cardboard boxes", "Books", "Gasoline engines"]
        for target in fire_targets_11: add_counter(target, "Fire")

        # Add Drought counters (List 12)
        drought_targets_12 = ["Rice paddies", "Corn fields", "Wheat farms", "Fruit orchards", "Vineyards", "Pastures for grazing", "Tropical rainforests", "Mangroves", "Wetlands", "Rivers", "Small lakes", "Fish ponds", "Coffee farms", "Cocoa plantations", "Palm groves", "Avocado orchards", "Reservoirs", "Water fountains in parks", "Golf courses", "Irrigated farmlands", "Dairy farms", "Desertification", "Water-hungry industries", "Cattle ranches", "Hydroponic farming", "Floriculture", "Aquaculture", "Water treatment plants", "Ecosystems dependent on fresh water", "Drinking water sources", "City water systems"]
        for target in drought_targets_12: add_counter(target, "Drought")

        # Add Water counters (List 13)
        water_targets_13 = ["Fire", "Drought", "Overheating machines", "Heatstroke", "Thirst", "Desertification", "Fire hazards", "Overcooked food", "Oil slicks", "Dry skin", "Dry eyes", "Dehydration", "Tiredness", "Hot beverages", "Chemical reactions", "Dust storms", "Airborne pollutants", "Gasoline fires", "Dry hair", "Extreme temperatures", "Stuck objects", "Fuel combustion", "Overcooked electronics", "Explosives", "Burns", "High heat environments", "Accumulated smoke particles", "Skin wounds", "Earthquake dust"]
        for target in water_targets_13: add_counter(target, "Water")

        # Add Vacuum counters (List 14)
        vacuum_targets_14 = ["Dust", "Spilled liquids", "Airborne bacteria", "Polluted air", "Small debris", "Carpet fibers", "Dirt on hard floors", "Pet hair", "Ash", "Stale air", "Clothing lint", "Old cobwebs", "Household odors", "Mold spores", "Lightweight plastics", "Small insects", "Broken glass shards", "Leaf litter", "Food crumbs", "Dust mites", "Pollen", "Smoke particles", "Microscopic dust", "Dry leaves", "Soot particles", "Carpet stains", "Static charge", "Small food scraps"]
        for target in vacuum_targets_14: add_counter(target, "Vacuum")

        # Add Laser counters (List 15)
        laser_targets_15 = ["Optical sensors", "Solar panels", "Drones", "Cameras", "Infrared devices", "Smoke detectors", "Light-based communication systems", "Satellites", "Security cameras", "Military infrared tracking", "Robot vision systems", "Solar-powered gadgets", "Power grids", "Light-sensitive materials", "Laser-guided weapons", "Optical fiber networks", "Laser cut jobs", "Night-vision equipment", "Thermal detectors", "Fire alarms", "Modern televisions", "Laser-based communication", "Targeted surveillance systems", "Photocells and photoelectric devices", "Sensors in medical devices", "Airborne targets", "Optical drive systems", "Printed barcodes"]
        for target in laser_targets_15: add_counter(target, "Laser")

        # Add Life Raft counters (List 16)
        life_raft_targets = ["Drowning", "Floodwaters", "Sinking ships", "River currents", "Tsunamis", "Capsized boats", "Hurricanes at sea", "Overboard accidents", "Coastal floods", "Rapid tides", "Ocean storms", "Broken boat engines", "Water rescues", "Flash floods", "Lake disasters", "Marooned survivors", "Water submersion", "Ice melt flooding", "Heavy rainfalls", "Rogue waves", "Overwhelmed ferries", "Leisure accidents", "Small plane water landings", "Collapsed docks", "Urban flooding", "Stranded cruise passengers", "Washed away vehicles", "Swamped islands", "Emergency evacuations", "Offshore oil rig disasters"]
        for target in life_raft_targets: add_counter(target, "Life Raft")

        # Add Bear Trap counters (List 17)
        bear_trap_targets = ["Intruding animals", "Wild bears", "Wolves", "Coyotes", "Foxes", "Cougars", "Mountain lions", "Predatory dogs", "Boars", "Leopards", "Bobcats", "Thieves", "Intruders", "Saboteurs", "Forest trespassers", "Jungle raiders", "Survival threats", "Park invaders", "Cunning prey", "Rogue robots", "Zombie limbs", "Alien creatures", "Mutant beasts", "Escaping prisoners", "Military sneaks", "Jungle survival hazards", "Dangerous fauna", "Stray cattle", "Escaping enemies", "Hostile explorers"]
        for target in bear_trap_targets: add_counter(target, "Bear Trap")

        # Add Hydraulic Jack counters (List 18)
        hydraulic_jack_targets = ["Collapsed vehicles", "Crushed car frames", "Overturned trucks", "Stuck doors", "Heavy debris", "Trapped accident victims", "Pinned limbs under heavy objects", "Collapsed tunnels", "Weighed-down structures", "Bridge collapses", "Crushed boats", "Tight machinery jams", "Earthquake rubble", "Car tire changes", "Submerged structures", "Collapsed elevators", "Industrial machinery jams", "Shipping container collapses", "Lift fallen logs", "Trapped miners", "Stuck train cars", "Compressed aircraft fuselage", "Heavy furniture lifts", "Shipping yard accidents", "Building collapses", "Tractor flips", "Construction equipment failures", "Subway derailments", "Large industrial gates stuck", "Trapped underground dwellers"]
        for target in hydraulic_jack_targets: add_counter(target, "Hydraulic Jack")

        # Add Diamond Cage counters (List 19)
        diamond_cage_targets = ["Explosive blasts", "Heavy physical attacks", "Wild animals", "Dragons", "Laser attacks", "Acid sprays", "Bullets", "Shrapnel", "Nuclear fallout debris", "Hulk-type strength creatures", "Crushing attacks", "Giant robots", "Electric surges", "Time bombs", "Zombies", "Rioting mobs", "Collapsing ceilings", "Powerful mutants", "EMP shocks", "Tsunamis", "Tornado winds", "Super-strength beings", "Avalanche debris", "Feral beasts", "Alien abductions", "Giant insects", "Prison breaks", "Urban warfare", "Terrorist bomb threats", "Trapped magical creatures"]
        for target in diamond_cage_targets: add_counter(target, "Diamond Cage")

        # Add Dam counters (List 20)
        dam_targets = ["Flooding", "Tsunamis", "River overflows", "Flash floods", "Spring thaws", "Melting glaciers", "Heavy rainfall", "Downstream surges", "Cropland erosion", "City flood threats", "Hydro energy failures", "Drought in adjacent fields", "Desertification", "Fish migrations", "Saline intrusions", "Irrigation needs", "Agricultural collapse", "Groundwater depletion", "Dry season farming", "Urban water supply shortage", "Electricity shortages", "Crop watering", "Drinking water scarcity", "Firefighting reserves", "Ecosystem imbalance", "Water tourism industry", "Dried rivers", "Rapid snow melts", "Mountain runoff", "Coastal flooding events"]
        for target in dam_targets: add_counter(target, "Dam")

        # Add Sunshine counters (List 21)
        sunshine_targets = ["Seasonal Affective Disorder", "Depression", "Mold growth", "Fungus spread", "Damp environments", "Chilly mornings", "Ice on roads", "Black ice", "Hypothermia", "Wet clothes drying", "Soggy fields", "Energy shortages", "Low vitamin D levels", "Sad moods", "Cold homes", "Weak crops", "Rotting leaves", "Fog", "Ice buildup on solar panels", "Dark cities", "Gloomy environments", "Snow on rooftops", "Frost on fields", "Winter sicknesses", "Mild cases of rickets", "Indoor humidity", "Frizzy hair", "Sleep disorders", "Weak immune systems", "Solar power deficits"]
        for target in sunshine_targets: add_counter(target, "Sunshine")

        # Add Mutation counters (List 22)
        mutation_targets = ["Static genomes", "Evolutionary stagnation", "Pathogen immunity", "Genetic diseases", "Weak species", "Vulnerability to environmental changes", "Inflexible ecosystems", "Endangered species extinction", "Weak immune responses", "Viral infections", "Cancer cell resistance", "Bacterial colonies", "Crop failures", "Pest invasions", "Soil salinity", "Drought-prone plants", "Susceptibility to plagues", "Environmental toxins", "Weakened DNA strands", "High UV radiation", "Predation pressures", "Hostile climates", "Diseases attacking old crops", "Insect plagues", "Ocean acidification resistance", "High temperatures", "Ice Age survival", "Rapid ecosystem changes", "Antibiotic failure scenarios"]
        for target in mutation_targets: add_counter(target, "Mutation")

        # Add Kevlar Vest counters (List 23)
        kevlar_vest_targets = ["Bullet impacts", "Knife stabs", "Shrapnel wounds", "Gunshots", "Bomb fragments", "Slashes from sharp objects", "Falling debris", "Low-caliber pistol shots", "Shotgun pellets", "Small explosions", "Machete attacks", "Broken glass", "Wild animal attacks", "Riot projectiles", "Rocks thrown", "Arrow strikes", "BB guns", "Spear attacks", "Pipe bomb shrapnel", "Grenade splinters", "Suicide bomber fragments", "Crossbow bolts", "Molotov shrapnel", "Improvised explosive devices debris", "Nail bombs", "Slashing attacks", "Axe blows", "Spiked club hits", "Construction debris"]
        for target in kevlar_vest_targets: add_counter(target, "Kevlar Vest")

        # Add Jackhammer counters (List 24)
        jackhammer_targets = ["Concrete floors", "Asphalt roads", "Stone walls", "Thick ice sheets", "Rock barriers", "Tree roots", "Brick walls", "Old pavement", "Sidewalks", "Ceramic tiles", "Stubborn foundations", "Frozen ground", "Stone statues", "Heavy sediment", "Mineral veins", "Coal seams", "Compacted soil", "Subterranean rocks", "Tunnel boring starts", "Mountain roads", "Riverbed rocks", "Dried clay", "Foundation pilings", "Construction waste", "Broken sewage pipes", "Dry concrete", "Ancient ruins", "Compacted sand layers", "Urban ruins", "Broken metro tunnels"]
        for target in jackhammer_targets: add_counter(target, "Jackhammer")

        # Add Signal Jammer counters (List 25)
        signal_jammer_targets = ["Drone control signals", "Radio communications", "Wi-Fi networks", "Bluetooth devices", "Cell phone signals", "Military transmissions", "GPS signals", "Satellite phone calls", "Emergency radio channels", "Internet routers", "Spy drones", "GPS tracking devices", "Hidden cameras", "Smart devices", "Wireless surveillance", "Bugged rooms", "Audio bugs", "Satellite drone feeds", "Automated vehicles", "Smartwatches", "Walkie-talkies", "IoT devices", "Military drones", "Enemy radar", "Police scanners", "Data exfiltration tools", "Wireless hacking attempts", "RFID systems", "Keyless car entry signals", "Home security alarms"]
        for target in signal_jammer_targets: add_counter(target, "Signal Jammer")


        # Validate and filter relationships - Keep only counters present in the word library
        validated_relationships = {}
        missing_counters = set()
        for keyword, counters in relationships.items():
            # Ensure counters list is unique
            unique_counters = list(dict.fromkeys(counters))
            valid_counters = []
            for counter in unique_counters:
                if counter in self.word_library:
                    valid_counters.append(counter)
                else:
                    missing_counters.add(counter)
            if valid_counters: # Only add keywords that have at least one valid counter
                 validated_relationships[keyword] = valid_counters

        if missing_counters:
            print("\nWarning: The following counter words used in relationships are not in the word library and were ignored:")
            print(", ".join(sorted(list(missing_counters))))
            print("-" * 80)

        return validated_relationships

    def _define_and_embed_categories(self):
        """Defines word categories and precomputes embeddings for their keywords."""
        print("Defining categories and embedding keywords...")
        categories = {
            "nature": ["forest", "tree", "plant", "river", "mountain", "ocean", "lake", "sea", "beach", "desert", "flower", "garden", "nature", "sand", "mud", "clay", "swamp", "jungle", "island", "canyon", "glacier"],
            "animal": ["animal", "bear", "lion", "tiger", "wolf", "shark", "predator", "creature", "beast", "dog", "cat", "bird", "monkey", "insect", "mammal", "fish", "mosquito", "rat", "termite", "locust"],
            "food": ["food", "fruit", "vegetable", "banana", "apple", "meat", "fish", "bread", "water", "drink"],
            "disaster": ["disaster", "catastrophe", "hurricane", "tornado", "earthquake", "flood", "drought", "tsunami", "volcano", "fire", "storm", "landslide", "avalanche", "wildfire", "sandstorm", "blizzard", "hail", "acid rain", "oil spill", "nuclear meltdown", "asteroid impact", "solar flare", "supernova", "gamma ray burst", "magnetic storm", "ice age", "sinkhole"],
            "technology": ["computer", "robot", "machine", "electronic", "device", "digital", "tech", "ai", "cyber", "phone", "laptop", "technology", "internet", "network", "software", "hardware", "engine", "motor", "drone", "satellite", "camera", "sensor", "laser", "battery", "generator", "cyberattack", "button", "zipper", "screw", "bolt", "nail", "key"],
            "weapon": ["weapon", "gun", "bomb", "missile", "explosive", "nuclear", "bullet", "armor", "tank", "fighter", "shield", "sword", "knife", "h-bomb", "dynamite", "molotov cocktail", "bear trap"],
            "transportation": ["car", "vehicle", "truck", "train", "plane", "boat", "ship", "submarine", "bus", "bicycle", "aircraft", "transport", "helicopter", "airplane", "drone"],
            "building": ["building", "structure", "wall", "tower", "skyscraper", "house", "castle", "fortress", "bridge", "dam", "door", "fence", "barrier", "pillar", "road", "sidewalk"],
            "human": [
                "human", "person", "people", "crowd", "individual", "citizen", "population", "civilization",
                "man", "woman", "boy", "girl", "adult", "teenager", "elder", "baby", "infant", "toddler", "youth", "senior",
                "family", "child", "children", "parent", "mother", "father", "sibling", "brother", "sister",
                "spouse", "husband", "wife", "relative", "ancestor", "descendant", "friend", "neighbor",
                "colleague", "partner", "community", "society", "resident", "inhabitant", "native", "foreigner", "stranger", "member",
                "volunteer", "participant", "observer", "worker", "employee", "employer", "professional", "amateur", "expert", "novice", "apprentice",
                "leader", "follower", "boss", "chief", "manager", "assistant", "secretary",
                "customer", "client", "patient", "student", "teacher", "professor", "researcher", "scientist",
                "doctor", "nurse", "engineer", "artist", "musician", "writer", "chef", "baker",
                "farmer", "builder", "carpenter", "electrician", "mechanic", "driver", "pilot", "sailor",
                "officer", "detective", "lawyer", "judge", "politician", "actor", "dancer", "singer",
                "athlete", "coach", "reporter", "journalist", "photographer", "designer", "architect",
                "accountant", "librarian", "priest", "monk", "nun", "barber", "hairdresser", "waiter",
                "waitress", "bartender", "receptionist", "janitor", "cleaner", "miner", "fisherman", "hunter",
                "explorer", "adventurer", "merchant", "trader", "shopkeeper", "technician", "programmer",
                "developer", "analyst", "consultant", "soldier", "guard", "spy", "agent", "assassin", "killer", "general", "admiral", "colonel",
                "major", "captain", "lieutenant", "sergeant", "corporal", "private", "king", "queen", "prince", "princess", "emperor", "duke", "duchess", "baron", "baroness",
                "lord", "lady", "sir", "madam", "president", "governor", "mayor", "senator", "representative", "ambassador", "pope", "bishop", "cardinal",
                "thief", "criminal", "pirate", "nomad", "settler", "refugee", "immigrant", "tourist", "traveler",
                "hero", "villain", "victim", "survivor", "witness", "enemy", "ally", "rival", "opponent", "competitor",
            ],
            "disease": ["disease", "virus", "bacteria", "infection", "pandemic", "plague", "epidemic", "pathogen", "mutation", "illness", "sickness", "cough", "frostbite", "hypothermia", "heatstroke"],
            "abstract": ["concept", "idea", "theory", "thought", "knowledge", "wisdom", "truth", "reality", "consciousness", "time", "love", "hate", "emotion", "fear", "anxiety", "depression", "belief", "movement", "progress", "change", "future", "past", "information", "data", "signal", "communication", "message", "news", "media", "chaos", "paradox", "mystery", "debt", "bureaucracy", "silence", "darkness", "ignorance", "sadness", "loneliness", "corruption", "tyranny", "misinformation", "stupidity", "apathy", "despair", "memory", "dream", "knowledge", "power outage", "traffic jam"],
            "broken heart": ["Love", "Therapy", "Time", "Vacation", "Confidence"], 
            "political unrest": ["Propaganda", "War", "Therapy", "Resilience", "Time", "Evacuation"], # Added political unrest
            "bad luck": ["Persistence", "Resilience", "Innovation", "Time"], # Added bad luck
            "writer's block": ["Innovation", "Persistence", "Confidence", "Vacation", "Therapy"], 
            "rusty nail": ["Acid", "Sandpaper", "Drill", "Oil", "Jackhammer"], # Added rusty nail
            "moldy bread": ["Fire", "Acid", "Vacuum", "Neutralizing Agent"], 
            "cosmic": ["space", "planet", "star", "universe", "galaxy", "asteroid", "meteor", "comet", "cosmic", "sun", "moon", "radiation", "black hole", "orbit", "satellite"],
            "supernatural": ["ghost", "spirit", "monster", "demon", "angel", "god", "deity", "magic", "supernatural", "mystical", "dragon", "vampire", "werewolf", "zombie", "spell"],
            "material": ["metal", "wood", "glass", "plastic", "stone", "rock", "steel", "gold", "silver", "earth", "sand", "concrete", "paper", "fabric", "leather", "paint", "glue", "rust", "mold", "iceberg", "ice", "cardboard", "thread", "string", "rope", "coin", "needle", "pin", "paperclip"],
            "elemental": ["fire", "water", "air", "earth", "lightning", "ice", "steam", "lava", "wind", "rain", "snow", "fog", "cloud", "heat", "cold"],
            "energy": ["energy", "power", "electricity", "nuclear", "battery", "generator", "solar", "wind power", "heat", "cold"],
            "defense": ["shield", "armor", "defense", "protection", "security", "lock", "encryption", "kevlar", "reinforced", "barrier", "fence", "wall", "cage"],
            "waste": ["shit", "poop", "waste", "garbage", "trash", "pollution", "toxic waste", "sewage", "junk", "debris", "stain", "dirt", "plastic bag"] # Added a waste category
        }
        category_keyword_embeddings = {}
        all_keywords = []
        keyword_to_category = {}

        for category, keywords in categories.items():
            unique_keywords = list(set(keywords)) # Ensure unique keywords per category
            categories[category] = unique_keywords # Update category with unique list
            for keyword in unique_keywords:
                if keyword not in keyword_to_category: # Avoid duplicate embeddings if keyword is in multiple categories
                    all_keywords.append(keyword)
                    keyword_to_category[keyword] = []
                keyword_to_category[keyword].append(category) # Map keyword back to its categories

        # Embed all unique keywords in batches
        print(f"Embedding {len(all_keywords)} unique category keywords...")
        all_embeddings = self.model.encode(all_keywords, batch_size=32, show_progress_bar=True)
        keyword_embedding_map = {keyword: embedding for keyword, embedding in zip(all_keywords, all_embeddings)}

        # Organize embeddings by category
        for category, keywords in categories.items():
            category_keyword_embeddings[category] = {
                "keywords": keywords,
                "embeddings": np.array([keyword_embedding_map[kw] for kw in keywords if kw in keyword_embedding_map])
            }
            if category_keyword_embeddings[category]["embeddings"].shape[0] == 0:
                 print(f"Warning: No valid embeddings found for category '{category}'.")


        print("Category keyword embeddings created.")
        return categories, category_keyword_embeddings
    
    def _get_sentence_embedding(self, sentence):
        """Get embedding for a sentence with caching, using the class model and device."""
        if sentence in self.embedding_cache:
            return self.embedding_cache[sentence]
        # Use torch.no_grad() for inference efficiency
        with torch.no_grad():
            # Encode directly to numpy array on CPU after inference on GPU/CPU
            embedding = self.model.encode(sentence, convert_to_tensor=False, device=self.device)
        self.embedding_cache[sentence] = embedding
        return embedding

    def _find_counters_semantic_search(self, system_word, top_n=10):
        """
        Find counter words based purely on semantic similarity using contextual embeddings.
        Adapted from the provided message.txt script.
        """
        print(f"Performing semantic search for counters against '{system_word}'...")
        query_contexts = [
            f"what counters {system_word}",
            f"what neutralizes {system_word}",
            f"what is effective against {system_word}",
            f"what stops {system_word}"
        ]
        # Get embeddings for query contexts
        query_embeddings = [self._get_sentence_embedding(context) for context in query_contexts]
        # Average the embeddings to get a single query vector
        avg_query_embedding = np.mean(query_embeddings, axis=0)

        counter_scores = []
        max_cost = max(data["cost"] for data in self.word_library.values()) if self.word_library else 100.0

        for word, data in self.word_library.items():
            if word.lower() == system_word.lower():
                continue # Skip the word itself

            # Create contexts for the candidate counter word
            candidate_contexts = [
                f"{word} counters {system_word}",
                f"{word} neutralizes {system_word}",
                f"{word} is effective against {system_word}",
                f"{word} stops {system_word}"
            ]
            # Get and average embeddings for candidate contexts
            candidate_embeddings = [self._get_sentence_embedding(context) for context in candidate_contexts]
            avg_candidate_embedding = np.mean(candidate_embeddings, axis=0)

            # Calculate cosine similarity (1 - cosine distance)
            # Handle potential NaN if embeddings are zero vectors
            similarity = 0.0
            try:
                cos_dist = cosine(avg_query_embedding, avg_candidate_embedding)
                if not np.isnan(cos_dist):
                    similarity = 1.0 - cos_dist
            except Exception: # Catch potential errors in cosine calculation
                 pass # Keep similarity at 0.0

            cost = data["cost"]
            # Calculate a combined score (similarity weighted more, cost penalty)
            # Normalize cost (0 to 1)
            normalized_cost = cost / max_cost if max_cost > 0 else 0.0
            # Adjust weights as needed (e.g., 0.7 for similarity, 0.3 penalty for cost)
            combined_score = (similarity * 0.7) - (normalized_cost * 0.3)

            counter_scores.append({
                "word": word,
                "cost": cost,
                "score": combined_score,
                "similarity": similarity, # This is the contextual similarity
                "thematic_score": similarity # Use similarity as thematic score for this method
            })

        # Sort by the combined score, highest first
        counter_scores.sort(key=lambda x: x["score"], reverse=True)
        print(f"Semantic search completed. Found {len(counter_scores)} potential counters.")
        return counter_scores[:top_n] # Return top N results

    def _calculate_semantic_similarity(self, system_word, counter_word):
        """Calculate direct semantic similarity between system word and counter word"""
        # Ensure counter_word exists in embeddings
        if counter_word not in self.counter_embeddings:
            print(f"Warning: Embedding not found for counter word '{counter_word}'.")
            return 0.0

        # Get embedding for system word (use caching method)
        system_embedding = self._get_sentence_embedding(system_word)

        # Get pre-calculated embedding for counter word
        counter_embedding = self.counter_embeddings[counter_word]

        # Calculate cosine similarity using sklearn's function for consistency
        # Reshape for cosine_similarity function
        similarity = cosine_similarity(system_embedding.reshape(1, -1), counter_embedding.reshape(1, -1))[0][0]

        # Clip similarity to be within [-1, 1]
        return np.clip(similarity, -1.0, 1.0)

    def _match_keywords(self, system_word):
        """Match keywords in the system word to find suitable counters"""
        system_word_lower = system_word.lower()

        # Direct matches from our knowledge base
        direct_matches = []
        match_scores = {} # Store scores for sorting

        for keyword, counters in self.word_relationships.items():
            keyword_lower = keyword.lower()
            score = 0
            if keyword_lower == system_word_lower: # Exact match gets highest score
                score = 2.0
            elif keyword_lower in system_word_lower: # Keyword is part of the input
                score = 1.5
            elif system_word_lower in keyword_lower: # Input is part of the keyword
                score = 1.0

            if score > 0:
                for counter in counters:
                    # Add score to existing counter score or initialize it
                    match_scores[counter] = match_scores.get(counter, 0) + score

        # If we found matches, sort them by score
        if match_scores:
            # Sort counters by their accumulated score, descending
            sorted_counters = sorted(match_scores.items(), key=lambda item: item[1], reverse=True)
            return [counter for counter, _ in sorted_counters] # Return only the counter names in order

        return [] # Return empty list if no matches found

    def _estimate_word_type(self, system_word):
        """Estimate the general type/category of the system word using semantic similarity."""
        system_word_lower = system_word.lower()
        min_similarity_threshold = 0.25 # Minimum average similarity to consider a category match

        # --- Stage 1: Direct Keyword Match (Fastest) ---
        matched_categories = []
        for category, keywords in self.categories.items():
            if system_word_lower in keywords:
                matched_categories.append(category)

        if matched_categories:
            # If it matches multiple, we might need a tie-breaker, but for now, let's pick the first.
            # Or, perhaps prioritize more specific categories if applicable.
            # For simplicity, returning the first match found.
            # print(f"Direct keyword match found for '{system_word}': Category '{matched_categories[0]}'") # Debugging
            return matched_categories[0]

        # --- Stage 2: Semantic Similarity Match (Slower) ---
        try:
            system_embedding = self.model.encode([system_word_lower])[0].reshape(1, -1)
        except Exception as e:
            print(f"Error encoding system word '{system_word_lower}': {e}")
            return "unknown" # Fallback if encoding fails

        category_scores = {}
        for category, data in self.category_keyword_embeddings.items():
            embeddings = data["embeddings"]
            if embeddings.shape[0] > 0: # Check if category has embeddings
                # Calculate cosine similarities between system word and all keywords in the category
                similarities = cosine_similarity(system_embedding, embeddings)[0]
                # Use the average similarity as the score for the category
                average_similarity = np.mean(similarities)
                category_scores[category] = average_similarity
            else:
                category_scores[category] = -1.0 # Assign low score if no embeddings

        if category_scores:
            # Find the category with the highest average similarity
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]

            # print(f"Similarity scores for '{system_word}': {category_scores}") # Debugging
            # print(f"Best matching category: '{best_category}' with score {best_score:.4f}") # Debugging

            # Only return the best category if its score meets the threshold
            if best_score >= min_similarity_threshold:
                return best_category
            else:
                # print(f"Best score {best_score:.4f} below threshold {min_similarity_threshold}. Falling back to 'unknown'.") # Debugging
                pass # Fall through to return "unknown"

        # Default to "unknown" if no category meets the threshold or if scoring fails
        return "unknown"

    def _get_fallback_counters(self, word_type):
        """Get fallback counters based on estimated word type"""
        # Added "waste" category and adjusted "unknown"
        fallbacks = {
            "nature": ["forest", "tree", "plant", "river", "mountain", "ocean", "lake", "sea", "beach", "desert", "flower", "garden", "nature", "sand", "mud", "clay", "swamp", "jungle", "island", "canyon", "glacier", "rapids", "geyser", "waterfall", "earth", "rock", "stone", "shifting sands", "rockslide"], 
             "animal": ["Bear Trap", "Dragon", "Diamond Cage", "Disease", "H-bomb", "Kevlar Vest", "Grizzly"],
            "dog": ["Bear Trap", "Love", "Therapy"],
            "cat": ["Love", "Therapy", "Water"],
            "bird": ["Gust", "Laser", "Signal Jammer"],
            "fish": ["Drought", "Acid", "Water"], # Added Water (e.g., for pollution)
            "insect": ["Fire", "Acid", "Vacuum", "Water"],
            "spider": ["Fire", "Acid", "Vacuum", "Boulder"],
            "snake": ["Bear Trap", "Fire", "Dragon", "Grizzly"],
            "venomous snake": ["Neutralizing Agent", "Kevlar Vest", "Dragon", "Fire", "Bear Trap", "Grizzly"], # Added venomous snake
            "bear": ["Bear Trap", "Dragon", "H-bomb", "Grizzly"],
            "wolf": ["Bear Trap", "Dragon", "Fire", "Grizzly"],
            "lion": ["Bear Trap", "Dragon", "H-bomb", "Grizzly"],
            "tiger": ["Bear Trap", "Dragon", "H-bomb", "Grizzly"],
            "shark": ["Dragon", "H-bomb", "Orbital Laser"],
            "whale": ["Dragon", "H-bomb", "Orbital Laser"], # Less aggressive, but large
            "rat": ["Bear Trap", "Fire", "Acid", "Disease"],
            "mosquito": ["Gust", "Vacuum", "Fire", "Disease"],
            "food": ["Acid", "Fire", "Disease", "Drought", "Freeze", "Water"],
            "disease": ["disease", "virus", "bacteria", "infection", "pandemic", "plague", "epidemic", "pathogen", "mutation", "illness", "sickness", "cough", "frostbite", "hypothermia", "heatstroke"],
            "disaster": ["disaster", "catastrophe", "hurricane", "tornado", "earthquake", "flood", "drought", "tsunami", "volcano", "fire", "storm", "landslide", "avalanche", "wildfire", "sandstorm", "blizzard", "hail", "acid rain", "oil spill", "nuclear meltdown", "asteroid impact", "solar flare", "supernova", "gamma ray burst", "magnetic storm", "ice age", "sinkhole", "tidal surge", "riptide current", "rockslide", "rumbling earth"], # Added more disaster types
            "nuclear winter": ["Terraforming Device", "Evacuation", "Cataclysm Containment Field", "Resilience", "Sunshine", "Time"], # Added nuclear winter
            "economic collapse": ["Innovation", "Resilience", "Persistence", "Propaganda", "Time"], # Added economic collapse
            "power outage": ["Innovation", "Resilience", "Time", "Dynamite"], # Added power outage (Dynamite for sabotage?)
            "wildfire smoke": ["Vacuum", "Gust", "Rainstorm", "Evacuation", "Water", "Fire Blanket"],
            "technology": ["Water", "Signal Jammer", "AI Kill Switch", "Acid", "Dynamite", "Laser", "Orbital Laser"],
            "weapon": ["Kevlar Vest", "Reinforced Steel Door", "Diamond Cage", "H-bomb", "Resilience", "Orbital Laser"],
            "transportation": ["Signal Jammer", "Dynamite", "Hurricane", "Laser", "Dragon", "War", "Explosion", "Steam", "Acid", "Fire"],
            "building": ["Earthquake", "Bulldozer", "Dynamite", "H-bomb", "Lava", "Fire", "Jackhammer", "Boulder"],
            "human": ["Disease", "Propaganda", "Love", "War", "Therapy", "Evacuation", "Resilience", "Viral Meme"],
            "disease": ["Anti-Virus Nanocloud", "Evacuation", "Neutralizing Agent", "Resilience", "Mutation", "Therapy"],
            "abstract": [
                # ... (existing abstract keywords) ...
                 "luck", "fate", "destiny", "mystery", "paradox", "persistence", "resilience", "confidence", "propaganda", "meme", "encryption", "broken heart", "bad luck", "writer's block", "economic collapse", "political unrest"
            ],
            "cosmic": ["Planetary Defense Shield", "Solar Deflection Array", "Antimatter Cannon", "Singularity Stabilizer", "Terraforming Device", "Orbital Laser"],
            "supernatural": ["Reality Resynchronizer", "Dragon", "H-bomb", "Singularity Stabilizer", "Time", "Vacuum", "Sunshine"],
            "material": ["Acid", "Laser", "Jackhammer", "Dynamite", "Fire", "Water", "Sandpaper", "Drill", "Bulldozer"],
            "elemental": ["Absorption", "Dam", "Freeze", "Vacuum", "Fire Extinguisher", "Seismic Dampener", "Cataclysm Containment Field", "Sunshine", "Steam"],
            "energy": ["Absorption", "Vacuum", "Singularity Stabilizer", "Water", "Acid", "Resilience", "Dam", "Signal Jammer"],
            "defense": ["Acid", "Laser", "Dynamite", "H-bomb", "Bulldozer", "Jackhammer", "Orbital Laser", "Reinforced Steel Door"],
            "waste": ["Vacuum", "Water", "Fire", "Acid", "Neutralizing Agent", "Absorption", "Bulldozer"], # Fallbacks for waste
            "unknown": ["Vacuum", "Water", "Acid", "Fire", "Bulldozer", "Resilience", "Innovation", "Time"] # Revised unknown fallbacks
        }

        # Ensure fallback counters exist in the library
        valid_counters = []
        if word_type in fallbacks:
            for counter in fallbacks[word_type]:
                if counter in self.word_library:
                    valid_counters.append(counter)
                # else: # Optional: Warn about missing fallbacks
                #     print(f"Warning: Fallback counter '{counter}' for type '{word_type}' not in library.")

        # If the specific type had no valid fallbacks, or type wasn't found, use the ultimate fallback
        if not valid_counters:
             # Use the specific "unknown" list if the type was explicitly determined as unknown
             fallback_key = "unknown" if word_type == "unknown" else "unknown" # Default to unknown list
             ultimate_fallback = fallbacks.get(fallback_key, ["Time", "Innovation", "Resilience"]) # Safely get unknown list
             valid_counters = [c for c in ultimate_fallback if c in self.word_library]

        return valid_counters

    def _select_best_counter(self, system_word):
        """Select the best counter word for a given system word"""
        potential_counters = []
        matched_counters = []
        is_direct_match = False
        word_type = "unknown" # Default

        # Step 1: Try direct keyword matching
        matched_counters = self._match_keywords(system_word)
        if matched_counters:
            is_direct_match = True
            word_type = "directly matched"
            print("Found direct relationship matches.")
        else:
            # Step 2: If no direct matches, estimate word type
            word_type = self._estimate_word_type(system_word)
            print(f"Estimated word type: {word_type}")
            if word_type != "unknown":
                # Use category-specific fallbacks
                matched_counters = self._get_fallback_counters(word_type)
            else:
                # Step 2b: If type is unknown, use the semantic search fallback
                # This directly returns the ranked list based on contextual similarity and cost
                potential_counters = self._find_counters_semantic_search(system_word, top_n=10)
                # If semantic search failed or returned nothing, use generic unknown fallbacks as last resort
                if not potential_counters:
                     print("Semantic search failed or yielded no results. Using generic 'unknown' fallbacks.")
                     matched_counters = self._get_fallback_counters("unknown")
                     word_type = "unknown_fallback" # Mark that we used generic fallbacks
                else:
                     # Semantic search succeeded, use its results directly
                     best_counter = potential_counters[0] if potential_counters else None
                     return {
                         "best": best_counter,
                         "type": "unknown (semantic search)", # Indicate semantic search was used
                         "top5": potential_counters[:5]
                     }

        # Step 3: Calculate scores ONLY if we didn't already return from semantic search
        # This part runs for direct matches or category fallbacks (non-unknown)

        # Calculate max_cost dynamically for normalization
        max_cost = max(data["cost"] for data in self.word_library.values()) if self.word_library else 100.0

        for counter_word in self.word_library:
            # Calculate direct similarity (word vs counter word)
            direct_similarity = self._calculate_semantic_similarity(system_word, counter_word)

            # Determine thematic relationship score based on keyword/fallback lists
            thematic_score = 0.0
            if counter_word in matched_counters:
                try:
                    position = matched_counters.index(counter_word)
                    if is_direct_match:
                        thematic_score = max(0, 1.0 - (position * 0.05))
                    else: # Category fallback match
                        thematic_score = max(0, 0.7 - (position * 0.08))
                except ValueError:
                    pass

            cost = self.word_library[counter_word]["cost"]
            cost_efficiency = 1.0 - (cost / max_cost) if max_cost > 0 else 0.0

            # Calculate final score with adjusted weights (Thematic > Similarity > Cost)
            final_score = (thematic_score * 0.65) + (direct_similarity * 0.20) + (cost_efficiency * 0.15)

            potential_counters.append({
                "word": counter_word,
                "cost": cost,
                "score": final_score,
                "similarity": direct_similarity, # Store direct similarity here
                "thematic_score": thematic_score
            })

        # Sort by final score (higher is better)
        potential_counters.sort(key=lambda x: x["score"], reverse=True)

        best_counter = potential_counters[0] if potential_counters else None

        return {
            "best": best_counter,
            "type": word_type, # Will be 'directly matched' or the estimated category name
            "top5": potential_counters[:5]
        }

    def find_counter(self, system_word):
        """Find the best counter for a given system word"""
        if not system_word or not system_word.strip():
             print("Error: Please provide a non-empty system word.")
             return None

        result = self._select_best_counter(system_word)
        best_counter = result["best"]
        top_counters = result["top5"]
        match_info = result["type"] # Contains estimated type or "directly matched"

        print(f"\nAnalyzing counters for: '{system_word}'")
        if match_info != "directly matched":
            print(f"Estimated word type: {match_info}")
        else:
            print("Found direct relationship matches.")

        if not best_counter:
            print("\nCould not determine a suitable counter.")
            return None

        print("\nBest counter word:")
        print("-" * 80)
        print(f"Word: {best_counter['word']}")
        print(f"Cost: {best_counter['cost']}")
        print(f"Thematic relationship score: {best_counter['thematic_score']:.4f}")
        print(f"Semantic similarity: {best_counter['similarity']:.4f}")
        print(f"Overall score: {best_counter['score']:.4f}")

        if top_counters:
            print("\nTop 5 counter options:")
            print("-" * 80)
            # Adjusted spacing for better alignment
            print(f"{'Word':<25} {'Cost':<8} {'Score':<10} {'Similarity':<15} {'Thematic':<15}")
            print("-" * 80)

            for counter in top_counters:
                print(f"{counter['word']:<25} {counter['cost']:<8} {counter['score']:.4f}    {counter['similarity']:.4f}       {counter['thematic_score']:.4f}")

        return best_counter['word']

def main():
    print("=== Words of Power Counter Analyzer ===")
    print("This tool will help you find the best counter words for the Words of Power game.")

    # Initialize game
    game = WordsOfPowerGame()
    # Initialize SpellChecker
    spell = SpellChecker()

    while True:
        print("\nEnter a system word to find its counter (or 'exit' to quit):")
        try:
            user_input = input("> ").strip() # Strip whitespace
        except EOFError: # Handle Ctrl+D or end of input stream
            print("\nGoodbye!")
            break

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if not user_input: # Check if input is empty after stripping
            print("Please enter a valid word.")
            continue

        # --- Add Spell Correction ---
        # Split input into words for potential multi-word inputs
        words = user_input.split()
        # Find potentially misspelled words
        misspelled = spell.unknown(words)
        corrected_words = []
        corrected_input = user_input # Default to original input
        made_correction = False

        if misspelled:
            corrected_list = []
            for word in words:
                if word in misspelled:
                    correction = spell.correction(word)
                    if correction != word:
                        corrected_list.append(correction)
                        made_correction = True
                    else:
                        corrected_list.append(word) # Keep original if correction is same or None
                else:
                    corrected_list.append(word)
            corrected_input = " ".join(corrected_list)

            if made_correction:
                 print(f"Corrected '{user_input}' to '{corrected_input}'")
        # --- End Spell Correction ---


        # Find and display the counter using the (potentially corrected) input
        game.find_counter(corrected_input)

if __name__ == "__main__":
    main()