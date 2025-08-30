import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import requests
from typing import List, Tuple, Dict, Set
import nltk
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class SpellingCorrector:
    def __init__(self):
        self.vocabulary = set()
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(lambda: defaultdict(Counter))
        self.word_vectors = {}
        self.tfidf_vectorizer = None
        self.corpus_text = ""
        self.homophones = self._load_homophones()
        self.confusables = self._load_confusables()
        self.medical_collocations = self._load_medical_collocations()
        
    def _load_homophones(self) -> Dict[str, Set[str]]:
        """Load common homophones and near-homophones in medical context"""
        return {
            'to': {'too', 'two'},
            'too': {'to', 'two'},
            'two': {'to', 'too'},
            'there': {'their', 'theyre'},
            'their': {'there', 'theyre'},
            'theyre': {'there', 'their'},
            'your': {'youre'},
            'youre': {'your'},
            'its': {'its'},
            'its': {'its'},
            'then': {'than'},
            'than': {'then'},
            'affect': {'effect'},
            'effect': {'affect'},
            'dose': {'does'},
            'does': {'dose'},
            'site': {'sight', 'cite'},
            'sight': {'site', 'cite'},
            'cite': {'site', 'sight'},
            'vein': {'vain'},
            'vain': {'vein'},
            'heel': {'heal'},
            'heal': {'heel'},
            'patients': {'patience'},
            'patience': {'patients'},
            'muscle': {'mussel'},
            'mussel': {'muscle'},
            'ileum': {'ilium'},
            'ilium': {'ileum'},
            'foul': {'fowl'},
            'fowl': {'foul'},
            'mucus': {'mucous'},
            'mucous': {'mucus'},
            'prostrate': {'prostate'},
            'prostate': {'prostrate'}
        }
    
    def _load_confusables(self) -> Dict[str, Set[str]]:
        """Load commonly confused words in medical contexts"""
        return {
            'except': {'accept'},
            'accept': {'except'},
            'advice': {'advise'},
            'advise': {'advice'},
            'loose': {'lose'},
            'lose': {'loose'},
            'breath': {'breathe'},
            'breathe': {'breath'},
            'quite': {'quiet'},
            'quiet': {'quite'},
            'complement': {'compliment'},
            'compliment': {'complement'},
            'principle': {'principal'},
            'principal': {'principle'},
            'stationary': {'stationery'},
            'stationery': {'stationary'},
            'discreet': {'discrete'},
            'discrete': {'discreet'},
            'elicit': {'illicit'},
            'illicit': {'elicit'},
            'eminent': {'imminent'},
            'imminent': {'eminent'},
            'palpation': {'palpitation'},
            'palpitation': {'palpation'},
            'perfusion': {'profusion'},
            'profusion': {'perfusion'},
            'perineal': {'peroneal'},
            'peroneal': {'perineal'}
        }
    
    def _load_medical_collocations(self) -> Dict[str, Dict[str, Set[str]]]:
        """Load common medical collocations and expected word patterns"""
        return {
            'quantifiers': {
                'tablets': {'two', 'three', 'four', 'five', 'six', 'ten', 'multiple', 'several'},
                'pills': {'two', 'three', 'four', 'five', 'six', 'ten', 'multiple', 'several'},
                'doses': {'two', 'three', 'four', 'five', 'multiple', 'several', 'daily'},
                'times': {'two', 'three', 'four', 'five', 'six', 'multiple', 'several'},
                'capsules': {'two', 'three', 'four', 'five', 'six', 'multiple'},
                'milligrams': {'five', 'ten', 'twenty', 'fifty', 'hundred'},
                'milliliters': {'five', 'ten', 'twenty', 'fifty', 'hundred'},
                'days': {'two', 'three', 'four', 'five', 'seven', 'ten', 'fourteen'},
                'weeks': {'two', 'three', 'four', 'six', 'eight', 'twelve'},
                'months': {'two', 'three', 'four', 'six', 'nine', 'twelve'}
            },
            'prepositions': {
                'administered': {'to', 'by', 'with', 'for'},
                'prescribed': {'to', 'for', 'by'},
                'allergic': {'to'},
                'sensitive': {'to'},
                'responded': {'to', 'with'},
                'complained': {'of', 'about'},
                'suffering': {'from'},
                'diagnosed': {'with'},
                'treated': {'with', 'for', 'by'},
                'examined': {'by', 'for'},
                'referred': {'to', 'by', 'for'},
                'admitted': {'to', 'for', 'with'},
                'discharged': {'from', 'to', 'after'},
                'history': {'of'},
                'risk': {'of', 'for'},
                'symptoms': {'of'},
                'signs': {'of'},
                'evidence': {'of', 'for'}
            },
            'medical_phrases': {
                'blood': {'pressure', 'sugar', 'test', 'count', 'type', 'glucose', 'levels', 'work', 'culture'},
                'heart': {'rate', 'attack', 'disease', 'failure', 'surgery', 'condition', 'rhythm'},
                'chest': {'pain', 'x-ray', 'wall', 'tube', 'discomfort', 'tightness'},
                'abdominal': {'pain', 'distension', 'tenderness', 'examination', 'surgery'},
                'vital': {'signs', 'organs'},
                'medical': {'history', 'record', 'condition', 'treatment', 'examination'},
                'physical': {'examination', 'therapy', 'condition', 'activity'},
                'surgical': {'procedure', 'intervention', 'history', 'site'},
                'allergic': {'reaction', 'response', 'rhinitis'},
                'adverse': {'reaction', 'event', 'effect', 'response'},
                'side': {'effect', 'effects'},
                'laboratory': {'results', 'values', 'tests', 'findings'},
                'diagnostic': {'test', 'procedure', 'imaging', 'criteria'}
            }
        }
        
    def load_corpus(self):
        """Load comprehensive medical corpus from MTSamples + PubMed API"""
        try:
            import requests
            corpus_words = []
        
            # ============ PART 1: Load MTSamples Dataset ============
            st.info("Loading medical transcriptions corpus...")
            mtsamples_loaded = False
        
            try:
                # First try to download MTSamples automatically from GitHub mirror
                url = "https://raw.githubusercontent.com/chandelsman/Medical-Text-Classification/master/data/mtsamples.csv"
                df = pd.read_csv(url)
            
                # Process medical transcriptions
                for idx, row in df.iterrows():
                    # Get transcription text
                    if pd.notna(row.get('transcription', '')):
                        text = str(row['transcription']).lower()
                        words = re.findall(r'\b[a-z]+\b', text)
                        corpus_words.extend(words)
                
                    # Add medical specialties
                    if pd.notna(row.get('medical_specialty', '')):
                        specialty = str(row['medical_specialty']).lower()
                        corpus_words.extend(specialty.replace('/', ' ').split())
                
                    # Add keywords
                    if pd.notna(row.get('keywords', '')):
                        keywords = str(row['keywords']).lower()
                        corpus_words.extend([k.strip() for k in keywords.split(',')])
                
                    # Add sample names (procedures)
                    if pd.notna(row.get('sample_name', '')):
                        sample = str(row['sample_name']).lower()
                        corpus_words.extend(sample.split())
            
                mtsamples_loaded = True
                st.success(f"✓ Loaded MTSamples: {len(set(corpus_words))} unique terms")
            
            except Exception as e1:
                # If download fails, try local file
                try:
                    df = pd.read_csv('mtsamples.csv')
                
                    for idx, row in df.iterrows():
                        if pd.notna(row.get('transcription', '')):
                            text = str(row['transcription']).lower()
                            words = re.findall(r'\b[a-z]+\b', text)
                            corpus_words.extend(words)
                    
                        if pd.notna(row.get('medical_specialty', '')):
                            specialty = str(row['medical_specialty']).lower()
                            corpus_words.extend(specialty.replace('/', ' ').split())
                    
                        if pd.notna(row.get('keywords', '')):
                            keywords = str(row['keywords']).lower()
                            corpus_words.extend([k.strip() for k in keywords.split(',')])
                
                    mtsamples_loaded = True
                    st.success(f"✓ Loaded local MTSamples: {len(set(corpus_words))} unique terms")
                
                except FileNotFoundError:
                    st.warning("MTSamples not found locally. Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
                    st.info("Continuing with PubMed data only...")
        
            # ============ PART 2: Load PubMed Medical Literature ============
            st.info("Fetching medical literature from PubMed...")
        
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
            # Comprehensive medical search terms
            medical_topics = [
                'clinical diagnosis', 'patient treatment', 'medical therapy',
                'diabetes mellitus', 'hypertension management', 'cancer treatment',
                'cardiovascular disease', 'infectious disease', 'neurological disorders',
                'pediatric medicine', 'surgical procedures', 'pharmacology drugs',
                'emergency medicine', 'internal medicine', 'psychiatry mental health',
                'orthopedic surgery', 'obstetrics gynecology', 'radiology imaging',
                'anesthesiology', 'pathology laboratory', 'dermatology skin',
                'ophthalmology eye', 'otolaryngology ENT', 'urology kidney',
                'gastroenterology digestive', 'endocrinology hormones', 'hematology blood',
                'immunology allergy', 'nephrology renal', 'pulmonology respiratory',
                'rheumatology arthritis', 'clinical trials', 'medical research'
            ]
        
            # Progress bar for PubMed loading
            progress_bar = st.progress(0)
            pubmed_words = []
        
            for idx, topic in enumerate(medical_topics):
                try:
                    # Search for articles
                    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={topic}&retmax=100&retmode=json"
                    search_response = requests.get(search_url, timeout=5)
                    search_data = search_response.json()
                
                    # Get PMIDs
                    id_list = search_data.get('esearchresult', {}).get('idlist', [])[:50]
                
                    if id_list:
                        # Fetch abstracts
                        ids_string = ','.join(id_list)
                        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids_string}&rettype=abstract"
                        fetch_response = requests.get(fetch_url, timeout=10)
                    
                        # Extract medical terms (3+ characters)
                        text = fetch_response.text.lower()
                        words = [w for w in re.findall(r'\b[a-z]+\b', text) if len(w) >= 3]
                        pubmed_words.extend(words)
                
                    # Update progress
                    progress_bar.progress((idx + 1) / len(medical_topics))
                
                except Exception:
                    continue
        
            progress_bar.empty()
            corpus_words.extend(pubmed_words)
            st.success(f"✓ Loaded PubMed: {len(set(pubmed_words))} unique terms")
        
            # ============ PART 3: Add Medical Terminology Database ============
            st.info("Adding medical terminology database...")
        
            # Download medical terms list
            try:
                medical_terms_url = "https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
                response = requests.get(medical_terms_url)
                medical_terms = response.text.lower().split('\n')
                medical_terms = [term.strip() for term in medical_terms if term.strip() and len(term.strip()) >= 3]
                corpus_words.extend(medical_terms * 10)  # Add with frequency weight
                st.success(f"✓ Added {len(medical_terms)} medical dictionary terms")
            except:
                pass
        
            # ============ PART 4: Add Core Medical Vocabulary ============
            # Essential medical terms to ensure coverage
            core_medical_vocab = """
            abdominal abdomen abnormal abscess absorption accident acidosis acne acute adenoma adhesion adipose admission adrenal adult adverse airway albumin alcohol allergy alopecia alzheimer ambulance amino amnesia amniotic amputation analgesia analgesic anaphylaxis anastomosis anatomy anemia anesthesia aneurysm angina angiogram angioplasty ankle anomaly anorexia antacid anterior antibiotic antibody antidepressant antigen antihistamine antimicrobial antipsychotic antiseptic antiviral anxiety aorta aortic appendectomy appendicitis appendix appetite arrhythmia arterial arteriosclerosis artery arthritis arthroscopy articulation artificial ascites aseptic aspiration aspirin assessment asthma asymptomatic ataxia atherosclerosis atrial atrium atrophy attack audiometry auditory auscultation autism autoimmune autonomic autopsy axial axis axon
        
            bacteria bacterial bacterium balance balloon bandage barium barrier basal baseline behavior benign beta bicarbonate bilateral bile biliary bilirubin biochemical biopsy bipolar birth bladder bleeding blind blood blurred body bone bowel brachial bradycardia brain brainstem branch breast breath breathing bronchial bronchitis bronchoscopy bronchospasm bronchus bruise buffer bulimia burn burning bursa bursitis bypass
        
            cachexia caesarean calcification calcium calculus caliber calorie cancer candidiasis cannula capacity capillary capsule carbohydrate carbon carcinogen carcinoma cardiac cardiomyopathy cardiopulmonary cardiovascular care caregiver caries carotid carpal cartilage case cast cataract catheter catheterization cauterization cavity cell cellular cellulitis center central cerebellar cerebellum cerebral cerebrospinal cerebrovascular cerebrum certification cervical cervix cessation chamber change channel characteristic charting check chemical chemotherapy chest childhood children chlamydia chloride cholecystectomy cholecystitis cholera cholesterol chronic circulation circulatory cirrhosis classification claudication clavicle clearance cleft client clinical clinic clitoris clone clonic closure clot clotting cluster coagulation cochlea code cognitive coil cold colic colitis collapse colon colonoscopy color colorectal colostomy colposcopy coma combination comfort common communicable communication community comparison compartment compensation complaint complement complete complex compliance complication component compound comprehensive compression computed concentration conception concussion condition condom conduction conductive congenital congestion congestive conjunctiva conjunctivitis connective conscious consciousness consent conservative consideration consolidation constant constipation constitutional constriction consultation consumption contact contagious contamination content context continence continuation continuous contour contraception contraceptive contractility contraction contracture contraindication contralateral contrast control controlled contusion conventional conversion convulsion coordination cope coping cord core cornea corneal coronary corpus correction correlation cortex cortical corticosteroid cortisol cosmetic costal cough counseling count course coverage crack crackle cramp cranial craniotomy craving creatine creatinine crepitus crisis criteria critical cross croup crown crucial cruciate crush crust crutch cryotherapy culture cumulative curative cure current curvature curve cushion custom cutaneous cutting cyanosis cycle cyclic cylinder cyst cystectomy cystic cystitis cystoscopy cytology cytomegalovirus cytoplasm cytotoxic
        
            daily damage data database date dead deaf deafness death debridement debris decay decubitus deep defecation defect defense deficiency deficit definitive deformity degeneration degenerative dehydration delay delayed deletion delirium delivery delta deltoid delusion dementia demyelination dendrite denial dense density dental dentist dentition denture dependence dependent depersonalization depolarization deposit depression deprivation depth derivative dermal dermatitis dermatology dermis descending description desensitization design desire destruction detachment detail detection deterioration determination detoxification development developmental deviated deviation device diabetes diabetic diagnosis diagnostic dialysis diameter diaphoresis diaphragm diaphragmatic diarrhea diastole diastolic diet dietary differential differentiation diffuse diffusion digestion digestive digital dilatation dilation dilator dimension diminished dioxide diphtheria diplopia direct direction disability disabled disc discharge discipline discomfort disconnection discontinuation discrete discrimination disease disinfectant disinfection disk dislocation disorder disorganized disorientation displacement disposal disruption dissection disseminated dissociation distal distance distention distortion distress distribution disturbance diuresis diuretic diverticula diverticulitis diverticulosis diverticulum divided division dizziness doctor document documentation domain dome domestic dominant dominance donation donor dopamine doppler dormant dorsal dorsiflexion dorsum dosage dose double doubt douche down drain drainage drawing dream dressing drift drill drinking drip drive drooling drop droplet drug drunk dual duct ductus dull duodenal duodenum duplex duplicate dura durable duration dust duty dwarfism dying dynamic dysfunction dyslexia dysmenorrhea dyspareunia dyspepsia dysphagia dysphasia dysplasia dyspnea dysrhythmia dystocia dystonia dystrophy dysuria
        
            ear early eating ecchymosis echocardiogram echocardiography eclampsia ectasia ectopic ectopy eczema edema edematous edge education effect effective effector efferent efficacy efficiency effort effusion eight elastic elasticity elbow elderly elective electric electrical electrocardiogram electrocardiography electrode electroencephalogram electroencephalography electrolyte electromyography electron electronic electrophysiology element elevation eligible elimination emaciation embolectomy embolism embolization embolus embryo embryonic emergency emesis emission emotion emotional empathy emphysema empiric empty empyema emulsification enable enamel encephalitis encephalopathy encoding encounter endemic endocarditis endocardium endocrine endocrinology endogenous endometrial endometriosis endometrium endorphin endoscope endoscopic endoscopy endothelial endothelium endotracheal endurance enema energy engagement engine enhancement enlargement enteral enteric enteritis enterocele enterocolitis enterostomy entrapment entry enucleation enuresis environment environmental enzyme eosinophil eosinophilia ependyma epicardium epicondyle epidemic epidemiology epidermal epidermis epidural epigastric epiglottis epilepsy epinephrine epiphyseal epiphysis episode episodic epispadias epistaxis epithelial epithelium equilibrium equipment equivalent erectile erection erosion error eruption erythema erythrocyte erythropoiesis erythropoietin escape eschar esophageal esophagitis esophagoscopy esophagus essential established ester estimate estrogen ether ethical ethics ethmoid etiology eupnea eustachian euthanasia evacuation evaluation evaporation evening event eversion evidence evoked exacerbation examination example excavation excess exchange excision excitation excitement excoriation excretion excursion exercise exertion exfoliation exhalation exhaustion exocrine exogenous exophthalmos exostosis exotoxin expansion expectancy expectant expectorant expectoration experience experiment experimental expiration expiratory explanation exploration exploratory explosion exposure expression extension extensive extensor extent external extracellular extracorporeal extraction extradural extraocular extrapyramidal extrasystole extrauterine extravasation extremity extrinsic exudate exudation eye eyeball eyelid
        
            face facial facilitate facility factor failure fainting fall fallopian false familial family fascia fasciculation fasciotomy fasting fatal fatigue fatty faucial fauces febrile fecal feces feeding feet fellow female femoral femur fenestration ferritin fertile fertility fertilization fetal fetus fever fiber fibrillation fibrin fibrinogen fibroblast fibroid fibroma fibrosis fibrous fibula field fifth figure filament film filter filtration fimbria final finding fine finger first fissure fistula fitness five fixation flaccid flagellum flank flap flat flatulence flatus flexibility flexion flexor flexure flight floating floor flora flow fluctuation fluid fluorescence fluoride fluoroscopy flush flutter foam focal focus fold foley follicle follicular fontanelle food foot foramen force forceps forearm foreign foreskin form formation formula fornix fossa four fovea fraction fracture fragile fragment frank free freedom fremitus frequency frequent friction frontal frostbite frozen fructose full function functional fundus fungal fungus funnel fusion
            """
        
            core_words = core_medical_vocab.split()
            corpus_words.extend(core_words * 20)  # Add core vocabulary with good frequency
        
            # ============ PART 5: Build Final Corpus ============
            self.word_freq = Counter(corpus_words)
            self.vocabulary = set(self.word_freq.keys())
        
            # Build bigrams and trigrams for context checking
            st.info("Building n-gram models...")
            for i in range(len(corpus_words) - 2):
                if i < len(corpus_words) - 1:
                    self.bigrams[corpus_words[i]][corpus_words[i + 1]] += 1
                if i < len(corpus_words) - 2:
                    self.trigrams[corpus_words[i]][corpus_words[i + 1]][corpus_words[i + 2]] += 1
        
            # Store sample text for context
            self.corpus_text = ' '.join(corpus_words[:100000])
        
            # Final statistics
            total_words = len(self.vocabulary)
            total_tokens = sum(self.word_freq.values())
        
            st.success(f"""
            ✅ **Corpus Successfully Loaded!**
            - Unique medical terms: **{total_words:,}**
            - Total word tokens: **{total_tokens:,}**
            - Bigram pairs: **{len(self.bigrams):,}**
            - Trigram pairs: **{sum(len(v) for v in self.trigrams.values()):,}**
            - Sources: MTSamples + PubMed + Medical Dictionary
            """)
        
            return True
        
        except Exception as e:
            st.error(f"Error loading corpus: {e}")
            return False    
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate minimum edit distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between two strings"""
        set1 = set(s1)
        set2 = set(s2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0

    def get_candidates(self, word: str, max_distance: int = 2) -> List[Tuple[str, int, float]]:
        """Generate candidate corrections with edit distance and frequency score"""
        candidates = []
        word_lower = word.lower()
        
        for vocab_word in self.vocabulary:
            distance = self.levenshtein_distance(word_lower, vocab_word)
            if distance <= max_distance:
                freq_score = self.word_freq[vocab_word] / sum(self.word_freq.values())
                jaccard_sim = self.jaccard_similarity(word_lower, vocab_word)
                # Combined score: lower distance is better, higher frequency is better
                combined_score = (1 / (distance + 1)) * freq_score * (jaccard_sim + 0.1)
                candidates.append((vocab_word, distance, combined_score))
        
        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:5]  # Return top 5 candidates

    def get_candidates_advanced(self, word: str, context: str = "", max_distance: int = 2) -> List[Tuple[str, Dict[str, float]]]:
        """
        Generate candidate corrections using ALL similarity methods:
        - Levenshtein Distance
        - Jaccard Similarity
        - Cosine Similarity (with context)
        - N-gram similarity
        - Bigram probability
        - Trigram probability (NEW)
        - Homophone/Confusable check (NEW)
        """
        candidates = []
        word_lower = word.lower()
        
        # Check if word is a homophone or confusable
        is_homophone = word_lower in self.homophones
        is_confusable = word_lower in self.confusables
        
        # Get potential candidates from different sources
        potential_candidates = []
        
        # 1. Edit distance candidates
        for vocab_word in self.vocabulary:
            distance = self.levenshtein_distance(word_lower, vocab_word)
            if distance <= max_distance:
                potential_candidates.append((vocab_word, distance))
        
        # 2. Add homophones if applicable
        if is_homophone:
            for homophone in self.homophones[word_lower]:
                if homophone in self.vocabulary:
                    distance = self.levenshtein_distance(word_lower, homophone)
                    potential_candidates.append((homophone, distance))
        
        # 3. Add confusables if applicable
        if is_confusable:
            for confusable in self.confusables[word_lower]:
                if confusable in self.vocabulary:
                    distance = self.levenshtein_distance(word_lower, confusable)
                    potential_candidates.append((confusable, distance))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for vocab_word, distance in potential_candidates:
            if vocab_word not in seen:
                seen.add(vocab_word)
                unique_candidates.append((vocab_word, distance))
        
        # Calculate all similarity metrics for each candidate
        for vocab_word, edit_distance in unique_candidates:
            metrics = {}
            
            # 1. Levenshtein Distance (normalized)
            metrics['levenshtein'] = 1.0 / (edit_distance + 1)
            
            # 2. Jaccard Similarity (character-level)
            metrics['jaccard'] = self.jaccard_similarity(word_lower, vocab_word)
            
            # 3. Frequency Score
            metrics['frequency'] = self.word_freq[vocab_word] / sum(self.word_freq.values())
            
            # 4. Cosine Similarity (if context provided)
            if context:
                context_with_candidate = context.replace(word, vocab_word)
                cos_sim = self.calculate_context_similarity(context, context_with_candidate)
                metrics['cosine'] = cos_sim
            else:
                metrics['cosine'] = 0.5
            
            # 5. N-gram character similarity
            metrics['ngram'] = self.ngram_similarity(word_lower, vocab_word)
            
            # 6. Bigram context probability
            bigram_score = 0.0
            if context:
                words = context.lower().split()
                word_idx = -1
                for i, w in enumerate(words):
                    if w == word_lower:
                        word_idx = i
                        break
                
                if word_idx > 0:
                    prev_word = words[word_idx - 1]
                    bigram_score = self.check_bigram_probability_smoothed(prev_word, vocab_word)
            metrics['bigram_prob'] = bigram_score
            
            # 7. Trigram context probability (NEW)
            trigram_score = 0.0
            if context and word_idx > 0:
                if word_idx > 1:
                    prev_prev_word = words[word_idx - 2]
                    prev_word = words[word_idx - 1]
                    trigram_score = self.check_trigram_probability(prev_prev_word, prev_word, vocab_word)
            metrics['trigram_prob'] = trigram_score
            
            # 8. Homophone/Confusable bonus (NEW)
            homophone_bonus = 0.0
            if is_homophone and vocab_word in self.homophones.get(word_lower, set()):
                homophone_bonus = 0.3
            if is_confusable and vocab_word in self.confusables.get(word_lower, set()):
                homophone_bonus = max(homophone_bonus, 0.25)
            metrics['homophone_bonus'] = homophone_bonus
            
            # 9. Collocation score (NEW)
            collocation_score = self.check_collocation_score(vocab_word, context) if context else 0.0
            metrics['collocation_score'] = collocation_score
            
            # Calculate combined score with weighted metrics
            combined_score = (
                metrics['levenshtein'] * 0.20 +
                metrics['jaccard'] * 0.15 +
                metrics['frequency'] * 0.10 +
                metrics['cosine'] * 0.10 +
                metrics['ngram'] * 0.10 +
                metrics['bigram_prob'] * 0.10 +
                metrics['trigram_prob'] * 0.10 +
                metrics['homophone_bonus'] * 0.10 +
                metrics['collocation_score'] * 0.05
            )
            
            metrics['combined_score'] = combined_score
            metrics['edit_distance'] = edit_distance
            
            candidates.append((vocab_word, metrics))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1]['combined_score'], reverse=True)
        return candidates[:10]

    def check_trigram_probability(self, word1: str, word2: str, word3: str) -> float:
        """Check trigram probability for better context modeling"""
        if word1 in self.trigrams and word2 in self.trigrams[word1]:
            total = sum(self.trigrams[word1][word2].values())
            if total > 0:
                count = self.trigrams[word1][word2][word3]
                # Add-one smoothing
                smoothed_prob = (count + 1) / (total + len(self.vocabulary))
                return smoothed_prob
        return 1 / (len(self.vocabulary) + 1)
    
    def check_collocation_score(self, word: str, context: str) -> float:
        """Check if word fits expected medical collocations"""
        words = context.lower().split()
        score = 0.0
        
        # Check quantifier collocations (e.g., "two tablets")
        for key, expected_words in self.medical_collocations['quantifiers'].items():
            if key in words:
                key_idx = words.index(key)
                if key_idx > 0 and words[key_idx - 1] == word:
                    if word in expected_words:
                        score += 0.8
                    else:
                        score -= 0.3
        
        # Check preposition collocations
        for key, expected_preps in self.medical_collocations['prepositions'].items():
            if key in words:
                key_idx = words.index(key)
                if key_idx < len(words) - 1 and words[key_idx + 1] == word:
                    if word in expected_preps:
                        score += 0.7
                    else:
                        score -= 0.2
        
        # Check medical phrase collocations
        for key, expected_follows in self.medical_collocations['medical_phrases'].items():
            if key in words:
                key_idx = words.index(key)
                if key_idx < len(words) - 1 and words[key_idx + 1] == word:
                    if word in expected_follows:
                        score += 0.6
        
        return min(max(score, 0.0), 1.0)  # Normalize to [0, 1]

    def ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        """
        Calculate n-gram similarity between two strings
        Useful for detecting phonetically similar words
        """
        def get_ngrams(s, n):
            return set([s[i:i+n] for i in range(len(s)-n+1)])
        
        if len(s1) < n or len(s2) < n:
            return 0.0
        
        ngrams1 = get_ngrams(s1, n)
        ngrams2 = get_ngrams(s2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0

    def calculate_context_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two text snippets using TF-IDF
        """
        try:
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.5

    def generate_corrected_text(self, text: str, confidence_threshold: float = 0.3) -> Tuple[str, List[Dict]]:
        """
        Generate automatically corrected text using highest scoring suggestions
        
        Args:
            text: Original text with errors
            confidence_threshold: Minimum combined score to accept a correction (0-1)
        
        Returns:
            Tuple of (corrected_text, corrections_made)
        """
        corrected_text = text
        corrections_made = []
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Track word positions for accurate replacement
        word_positions = []
        current_pos = 0
        for word in words:
            start = text.lower().find(word.lower(), current_pos)
            if start != -1:
                word_positions.append((word, start, start + len(word)))
                current_pos = start + len(word)
        
        # Process non-word errors
        for word in words:
            if word.lower() not in self.vocabulary:
                # Get advanced candidates with all metrics
                candidates = self.get_candidates_advanced(word, context=text)
                
                if candidates and len(candidates) > 0:
                    best_candidate = candidates[0]
                    best_word = best_candidate[0]
                    best_score = best_candidate[1]['combined_score']
                    
                    # Only correct if confidence is above threshold
                    if best_score >= confidence_threshold:
                        # Replace all occurrences of the error word (case-insensitive)
                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                        corrected_text = pattern.sub(best_word, corrected_text)
                        
                        corrections_made.append({
                            'original': word,
                            'correction': best_word,
                            'confidence': best_score,
                            'edit_distance': best_candidate[1]['edit_distance'],
                            'type': 'non-word'
                        })
        
        # Process real-word errors (context-based) with enhanced detection
        real_word_errors = self.detect_real_word_errors_enhanced(text)
        for error_word, position, alternatives in real_word_errors:
            if alternatives:
                best_alternative = alternatives[0]
                
                # Calculate confidence for real-word correction
                candidates = self.get_candidates_advanced(error_word, context=text)
                confidence = 0.5  # Default confidence
                
                for candidate_word, metrics in candidates:
                    if candidate_word == best_alternative:
                        confidence = metrics['combined_score']
                        break
                
                if confidence >= confidence_threshold * 0.8:  # Slightly lower threshold for context errors
                    # Replace the specific occurrence based on position
                    words_in_text = corrected_text.split()
                    if position < len(words_in_text):
                        words_in_text[position] = best_alternative
                        corrected_text = ' '.join(words_in_text)
                        
                        corrections_made.append({
                            'original': error_word,
                            'correction': best_alternative,
                            'confidence': confidence,
                            'position': position,
                            'type': 'context'
                        })
        
        return corrected_text, corrections_made
    
    def check_bigram_probability(self, prev_word: str, word: str) -> float:
        """Check bigram probability for context-based correction"""
        if prev_word in self.bigrams:
            total = sum(self.bigrams[prev_word].values())
            if total > 0:
                return self.bigrams[prev_word][word] / total
        return 0.0
    
    def check_bigram_probability_smoothed(self, prev_word: str, word: str) -> float:
        """
        Improved bigram probability with smoothing for sparse data
        """
        if prev_word in self.bigrams:
            total = sum(self.bigrams[prev_word].values())
            if total > 0:
                count = self.bigrams[prev_word][word]
                # Add-one smoothing to handle unseen bigrams
                smoothed_prob = (count + 1) / (total + len(self.vocabulary))
                return smoothed_prob
        
        # Return small but non-zero probability for unseen bigrams
        return 1 / (len(self.vocabulary) + 1)

    def detect_homophone_errors(self, text: str) -> List[Tuple[str, int, List[str]]]:
        """
        Detect homophone and confusable word errors based on context
        Example: "too tablets" -> "two tablets"
        """
        words = text.lower().split()
        errors = []
        
        for i, word in enumerate(words):
            # Check if word is a homophone
            if word in self.homophones:
                possible_corrections = self.homophones[word]
                
                # Evaluate each possible correction in context
                best_alternatives = []
                for alternative in possible_corrections:
                    if alternative not in self.vocabulary:
                        continue
                    
                    # Score based on context
                    score = 0.0
                    
                    # Check bigram probability
                    if i > 0:
                        prev_word = words[i-1]
                        orig_bigram_prob = self.check_bigram_probability_smoothed(prev_word, word)
                        alt_bigram_prob = self.check_bigram_probability_smoothed(prev_word, alternative)
                        if alt_bigram_prob > orig_bigram_prob * 2:
                            score += 0.4
                    
                    # Check trigram probability
                    if i > 1:
                        prev_prev = words[i-2]
                        prev_word = words[i-1]
                        orig_trigram_prob = self.check_trigram_probability(prev_prev, prev_word, word)
                        alt_trigram_prob = self.check_trigram_probability(prev_prev, prev_word, alternative)
                        if alt_trigram_prob > orig_trigram_prob * 2:
                            score += 0.3
                    
                    # Check medical collocations
                    context_score = self.check_collocation_score(alternative, text)
                    if context_score > 0.5:
                        score += context_score * 0.3
                    
                    # Special case for quantifiers before medical units
                    if i < len(words) - 1:
                        next_word = words[i+1]
                        if next_word in self.medical_collocations['quantifiers']:
                            if alternative in self.medical_collocations['quantifiers'][next_word]:
                                score += 0.5
                    
                    if score > 0.3:
                        best_alternatives.append((alternative, score))
                
                # Sort by score and add to errors if we found better alternatives
                if best_alternatives:
                    best_alternatives.sort(key=lambda x: x[1], reverse=True)
                    errors.append((word, i, [alt[0] for alt in best_alternatives[:3]]))
            
            # Check confusables similarly
            if word in self.confusables:
                possible_corrections = self.confusables[word]
                best_alternatives = []
                
                for alternative in possible_corrections:
                    if alternative not in self.vocabulary:
                        continue
                    
                    score = 0.0
                    
                    # Context-based scoring
                    if i > 0:
                        prev_word = words[i-1]
                        orig_bigram_prob = self.check_bigram_probability_smoothed(prev_word, word)
                        alt_bigram_prob = self.check_bigram_probability_smoothed(prev_word, alternative)
                        if alt_bigram_prob > orig_bigram_prob * 1.5:
                            score += 0.3
                    
                    # Check if alternative fits better in medical context
                    context_score = self.check_collocation_score(alternative, text)
                    if context_score > 0.4:
                        score += context_score * 0.4
                    
                    if score > 0.25:
                        best_alternatives.append((alternative, score))
                
                if best_alternatives:
                    best_alternatives.sort(key=lambda x: x[1], reverse=True)
                    # Only add if not already detected as homophone error
                    if not any(err[0] == word and err[1] == i for err in errors):
                        errors.append((word, i, [alt[0] for alt in best_alternatives[:3]]))
        
        return errors

    def detect_real_word_errors_enhanced(self, text: str) -> List[Tuple[str, int, List[str]]]:
        """
        Enhanced real-word error detection combining all methods including homophones
        """
        # Get errors from different detection methods
        homophone_errors = self.detect_homophone_errors(text)
        medical_errors = self.detect_medical_context_errors(text)
        improved_errors = self.detect_real_word_errors_improved(text)
        
        # Combine and deduplicate errors
        all_errors = {}
        
        # Priority: homophone errors > medical errors > improved errors
        for word, position, alternatives in homophone_errors:
            key = (word, position)
            all_errors[key] = set(alternatives)
        
        for word, position, alternatives in medical_errors:
            key = (word, position)
            if key not in all_errors:
                all_errors[key] = set(alternatives)
            else:
                all_errors[key].update(alternatives)
        
        for word, position, alternatives in improved_errors:
            key = (word, position)
            if key not in all_errors:
                all_errors[key] = set(alternatives)
            else:
                all_errors[key].update(alternatives)
        
        # Convert back to list format
        final_errors = []
        for (word, position), alternatives in all_errors.items():
            final_errors.append((word, position, list(alternatives)[:3]))
        
        return final_errors

    def detect_real_word_errors_improved(self, text: str) -> List[Tuple[str, int, List[str]]]:
        """
        Improved real-word error detection using multiple context methods
        """
        words = text.lower().split()
        errors = []
        
        for i in range(len(words)):
            curr_word = words[i]
            
            # Only check words that exist in vocabulary
            if curr_word not in self.vocabulary:
                continue
                
            # Method 1: Bigram probability check (with relaxed thresholds)
            bigram_suspicious = False
            if i > 0:
                prev_word = words[i-1]
                bigram_prob = self.check_bigram_probability(prev_word, curr_word)
                
                # More relaxed thresholds
                if bigram_prob < 0.01 and self.word_freq[curr_word] < 1000:
                    bigram_suspicious = True
            
            # Method 2: Frequency-based suspicion
            freq_suspicious = False
            if self.word_freq[curr_word] < 50:  # Very rare words might be errors
                freq_suspicious = True
            
            # Method 3: Context similarity check
            context_suspicious = False
            if len(words) > 2:  # Need sufficient context
                # Create context without the suspicious word
                context_without_word = ' '.join(words[:i] + [''] + words[i+1:])
                
                # Find similar words and check if they fit better
                candidates = self.get_candidates(curr_word, max_distance=1)
                for candidate_word, _, _ in candidates[:3]:
                    context_with_candidate = ' '.join(words[:i] + [candidate_word] + words[i+1:])
                    
                    # Check if candidate fits better contextually
                    try:
                        original_sim = self.calculate_context_similarity(text, text)  # baseline
                        candidate_sim = self.calculate_context_similarity(text, context_with_candidate)
                        
                        if candidate_sim > original_sim * 1.1:  # 10% improvement
                            context_suspicious = True
                            break
                    except:
                        pass
            
            # Method 4: Medical domain specific checks
            medical_suspicious = False
            if i > 0 and i < len(words) - 1:
                prev_word = words[i-1]
                next_word = words[i+1]
                
                # Check for common medical patterns
                medical_patterns = {
                    'patient': ['has', 'was', 'is', 'presented', 'complained', 'suffered'],
                    'diagnosis': ['of', 'is', 'was', 'includes', 'shows'],
                    'treatment': ['for', 'of', 'includes', 'with', 'using'],
                    'symptoms': ['of', 'include', 'are', 'were', 'such'],
                }
                
                for pattern_word, expected_next in medical_patterns.items():
                    if prev_word == pattern_word and curr_word not in expected_next:
                        # Check if current word could be a misspelling of expected words
                        for expected in expected_next:
                            if self.levenshtein_distance(curr_word, expected) <= 2:
                                medical_suspicious = True
                                break
            
            # Combine suspicion signals
            if bigram_suspicious or (freq_suspicious and context_suspicious) or medical_suspicious:
                # Find better alternatives
                alternatives = []
                
                # Get candidates with better context fit
                candidates = self.get_candidates_advanced(curr_word, context=text, max_distance=2)
                
                for candidate_word, metrics in candidates[:5]:
                    # Check if candidate has better bigram probability
                    better_bigram = False
                    if i > 0:
                        prev_word = words[i-1]
                        candidate_bigram_prob = self.check_bigram_probability(prev_word, candidate_word)
                        original_bigram_prob = self.check_bigram_probability(prev_word, curr_word)
                        
                        if candidate_bigram_prob > original_bigram_prob * 2:  # Significantly better
                            better_bigram = True
                    
                    # Check frequency advantage
                    better_frequency = self.word_freq[candidate_word] > self.word_freq[curr_word] * 2
                    
                    # Check overall metrics
                    good_metrics = metrics['combined_score'] > 0.3
                    
                    if better_bigram or better_frequency or good_metrics:
                        confidence_score = (
                            metrics['combined_score'] * 0.4 +
                            (candidate_bigram_prob if i > 0 else 0.5) * 0.3 +
                            (self.word_freq[candidate_word] / sum(self.word_freq.values())) * 1000 * 0.3
                        )
                        alternatives.append((candidate_word, confidence_score))
                
                # Sort alternatives by confidence
                alternatives.sort(key=lambda x: x[1], reverse=True)
                
                if alternatives:
                    errors.append((curr_word, i, [alt[0] for alt in alternatives[:3]]))
        
        return errors

    def detect_medical_context_errors(self, text: str) -> List[Tuple[str, int, List[str]]]:
        """
        Medical domain-specific context error detection
        """
        words = text.lower().split()
        errors = []
        
        # Define medical context patterns
        medical_contexts = {
            'symptoms': {
                'preceding': ['patient', 'symptoms', 'complaint', 'presenting'],
                'following': ['include', 'are', 'were', 'of', 'such', 'like'],
                'common_errors': {
                    'pain': ['pan', 'pian', 'payn'],
                    'nausea': ['nasea', 'nausua', 'nausia'],
                    'fever': ['fver', 'fevr', 'feber'],
                    'fatigue': ['fatigue', 'fatige', 'fatique']
                }
            },
            'treatments': {
                'preceding': ['prescribed', 'treatment', 'therapy', 'medication'],
                'following': ['for', 'to', 'with', 'of', 'including'],
                'common_errors': {
                    'aspirin': ['aspirn', 'asprin', 'aspiran'],
                    'insulin': ['insullin', 'insuln', 'inslin'],
                    'antibiotic': ['antibiotic', 'antibotic', 'antibiutic']
                }
            },
            'anatomy': {
                'preceding': ['in', 'the', 'patient', 'examination'],
                'following': ['shows', 'reveals', 'indicates', 'is', 'was'],
                'common_errors': {
                    'abdomen': ['abdomen', 'abdomin', 'abdoman'],
                    'thorax': ['thorax', 'thoracs', 'thorex'],
                    'extremities': ['extremitis', 'extremeties', 'extrimities']
                }
            }
        }
        
        for i, word in enumerate(words):
            if word in self.vocabulary:
                # Check each medical context
                for context_type, patterns in medical_contexts.items():
                    # Check if word appears in medical context
                    context_match = False
                    
                    # Check preceding context
                    if i > 0 and words[i-1] in patterns['preceding']:
                        context_match = True
                    
                    # Check following context
                    if i < len(words) - 1 and words[i+1] in patterns['following']:
                        context_match = True
                    
                    if context_match:
                        # Check against common medical errors
                        for correct_word, error_variants in patterns['common_errors'].items():
                            if word in error_variants and word != correct_word:
                                errors.append((word, i, [correct_word]))
                            elif self.levenshtein_distance(word, correct_word) <= 2:
                                # Potential error if very similar to common medical term
                                if self.word_freq[correct_word] > self.word_freq[word] * 5:
                                    errors.append((word, i, [correct_word]))
        
        return errors

    def generate_possible_corrections(self, text: str, num_variations: int = 3) -> List[Dict[str, any]]:
        """
        Generate multiple possible corrected sentences with confidence scores and explanations
        
        Args:
            text: Original text with potential errors
            num_variations: Number of correction variations to generate (default 3)
        
        Returns:
            List of dictionaries containing:
                - corrected_text: The corrected sentence
                - confidence: Overall confidence score
                - corrections: List of individual corrections made
                - explanation: Human-readable explanation of changes
        """
        possible_corrections = []
        
        # First, identify all errors
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Find non-word errors
        non_word_errors = {}
        for word in words:
            if word.lower() not in self.vocabulary:
                candidates = self.get_candidates_advanced(word, context=text)
                if candidates:
                    non_word_errors[word] = candidates[:3]  # Top 3 for each error
        
        # Find real-word errors
        real_word_errors = self.detect_real_word_errors_enhanced(text)
        
        # Generate variations based on different confidence thresholds and combinations
        thresholds = [0.25, 0.35, 0.45]  # Different confidence levels
        
        for threshold_idx, threshold in enumerate(thresholds):
            corrected = text
            corrections_made = []
            total_confidence = 0.0
            explanation_parts = []
            
            # Process non-word errors
            for error_word, candidates in non_word_errors.items():
                if candidates:
                    # Select candidate based on threshold strategy
                    if threshold_idx == 0:  # Most aggressive
                        selected = candidates[0]
                    elif threshold_idx == 1 and len(candidates) > 1:  # Moderate
                        selected = candidates[1] if candidates[1][1]['combined_score'] > threshold else candidates[0]
                    else:  # Conservative
                        selected = None
                        for candidate in candidates:
                            if candidate[1]['combined_score'] > threshold:
                                selected = candidate
                                break
                    
                    if selected:
                        best_word = selected[0]
                        confidence = selected[1]['combined_score']
                        
                        if confidence >= threshold:
                            # Replace in text
                            pattern = re.compile(re.escape(error_word), re.IGNORECASE)
                            corrected = pattern.sub(best_word, corrected)
                            
                            corrections_made.append({
                                'original': error_word,
                                'correction': best_word,
                                'confidence': confidence,
                                'type': 'spelling'
                            })
                            
                            total_confidence += confidence
                            
                            # Add to explanation
                            explanation_parts.append(
                                f"'{error_word}' → '{best_word}' (spelling error, {confidence:.0%} confidence)"
                            )
            
            # Process real-word errors
            for error_word, position, alternatives in real_word_errors:
                if alternatives and position < len(corrected.split()):
                    # Determine selection strategy based on threshold
                    selected_alternative = None
                    
                    if threshold_idx == 0 and alternatives:  # Aggressive
                        selected_alternative = alternatives[0]
                    elif threshold_idx == 1 and len(alternatives) > 1:  # Moderate
                        selected_alternative = alternatives[0]
                    elif threshold_idx == 2 and alternatives:  # Conservative
                        # Only select if it's a clear homophone/confusable error
                        if error_word.lower() in self.homophones or error_word.lower() in self.confusables:
                            selected_alternative = alternatives[0]
                    
                    if selected_alternative:
                        words_in_corrected = corrected.split()
                        if position < len(words_in_corrected) and words_in_corrected[position].lower() == error_word.lower():
                            original_word = words_in_corrected[position]
                            words_in_corrected[position] = selected_alternative
                            corrected = ' '.join(words_in_corrected)
                            
                            # Determine error type
                            error_type = 'context'
                            if error_word.lower() in self.homophones:
                                error_type = 'homophone'
                            elif error_word.lower() in self.confusables:
                                error_type = 'confusable'
                            
                            corrections_made.append({
                                'original': original_word,
                                'correction': selected_alternative,
                                'position': position,
                                'type': error_type
                            })
                            
                            confidence = 0.7 if error_type in ['homophone', 'confusable'] else 0.5
                            total_confidence += confidence
                            
                            # Add to explanation
                            if error_type == 'homophone':
                                explanation_parts.append(
                                    f"'{original_word}' → '{selected_alternative}' (homophone error - sounds similar but different meaning)"
                                )
                            elif error_type == 'confusable':
                                explanation_parts.append(
                                    f"'{original_word}' → '{selected_alternative}' (commonly confused word)"
                                )
                            else:
                                explanation_parts.append(
                                    f"'{original_word}' → '{selected_alternative}' (context suggests this word fits better)"
                                )
            
            # Calculate overall confidence
            if corrections_made:
                avg_confidence = total_confidence / len(corrections_made)
                
                # Create explanation
                if explanation_parts:
                    explanation = "Corrections made: " + "; ".join(explanation_parts)
                else:
                    explanation = "No corrections needed"
                
                # Determine correction strategy label
                strategy_labels = ['Comprehensive', 'Balanced', 'Conservative']
                strategy_label = strategy_labels[threshold_idx]
                
                possible_corrections.append({
                    'corrected_text': corrected,
                    'confidence': avg_confidence,
                    'corrections': corrections_made,
                    'explanation': explanation,
                    'strategy': strategy_label,
                    'num_changes': len(corrections_made)
                })
        
        # Sort by confidence and remove duplicates
        seen_texts = set()
        unique_corrections = []
        for correction in sorted(possible_corrections, key=lambda x: x['confidence'], reverse=True):
            if correction['corrected_text'] not in seen_texts:
                seen_texts.add(correction['corrected_text'])
                unique_corrections.append(correction)
        
        # If no corrections were made, add the original as a possibility
        if not unique_corrections or all(c['num_changes'] == 0 for c in unique_corrections):
            unique_corrections = [{
                'corrected_text': text,
                'confidence': 1.0,
                'corrections': [],
                'explanation': 'No errors detected - text appears correct',
                'strategy': 'Original',
                'num_changes': 0
            }]
        
        return unique_corrections[:num_variations]

def create_gui():
    st.set_page_config(page_title="Advanced Medical Spelling Corrector", layout="wide")
    
    # Initialize session state
    if 'corrector' not in st.session_state:
        st.session_state.corrector = SpellingCorrector()
        st.session_state.corpus_loaded = False
    
    if 'selected_word' not in st.session_state:
        st.session_state.selected_word = None
    
    st.title("🏥 Advanced Medical Spelling Correction System")
    st.markdown("---")
    
    # Sidebar for corpus management
    with st.sidebar:
        st.header("📚 Corpus Management")
        
        if not st.session_state.corpus_loaded:
            if st.button("Load Medical Corpus", type="primary", help="Load combined MTSamples + PubMed medical corpus"):
                with st.spinner("Loading medical corpus..."):
                    if st.session_state.corrector.load_corpus():
                        st.session_state.corpus_loaded = True
                        st.success(f"Loaded {len(st.session_state.corrector.vocabulary)} unique medical terms!")
                        st.rerun()
        else:
            st.success(f"✓ Medical corpus loaded: {len(st.session_state.corrector.vocabulary)} words")
            
            # Dictionary viewer
            st.header("📖 Medical Dictionary")
            
            # Search functionality
            search_term = st.text_input("Search medical term:", key="search", placeholder="e.g., diabetes, anesthesia")
            
            if search_term:
                search_lower = search_term.lower()
                if search_lower in st.session_state.corrector.vocabulary:
                    st.success(f"✓ '{search_term}' found in dictionary")
                    freq = st.session_state.corrector.word_freq.get(search_lower, 0)
                    st.info(f"Frequency: {freq}")
                else:
                    st.warning(f"'{search_term}' not found")
                    # Show suggestions
                    candidates = st.session_state.corrector.get_candidates(search_term, max_distance=3)
                    if candidates:
                        st.write("Did you mean:")
                        for word, dist, score in candidates[:3]:
                            st.write(f"• {word}")
            
            # Display vocabulary
            if st.checkbox("Browse all medical terms"):
                sorted_words = sorted(list(st.session_state.corrector.vocabulary))
                
                # Filter options
                filter_letter = st.selectbox("Filter by first letter:", 
                                            ["All"] + list(string.ascii_lowercase))
                
                if filter_letter != "All":
                    sorted_words = [w for w in sorted_words if w.startswith(filter_letter)]
                
                # Pagination
                words_per_page = 100
                total_pages = max(1, len(sorted_words) // words_per_page + 1)
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                start_idx = (page - 1) * words_per_page
                end_idx = min(start_idx + words_per_page, len(sorted_words))
                
                st.write(f"Showing words {start_idx+1} to {end_idx} of {len(sorted_words)}")
                
                # Display words in columns
                cols = st.columns(4)
                page_words = sorted_words[start_idx:end_idx]
                for i, word in enumerate(page_words):
                    cols[i % 4].write(word)
            
            # Word frequency stats
            if st.checkbox("Show top frequent medical terms"):
                top_words = st.session_state.corrector.word_freq.most_common(30)
                df = pd.DataFrame(top_words, columns=["Medical Term", "Frequency"])
                st.dataframe(df, use_container_width=True)
    
    # Main content area
    if st.session_state.corpus_loaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("✍️ Medical Text Editor")
            
            # Add sample text button with more examples
            sample_texts = {
                "Sample 1 - Common Errors": "The patiant presented with symptms of diabetis including frequent urinaton and increased thrist. The docter prescribed insullin for better glucos control.",
                "Sample 2 - Homophone Errors": "The patient was given too tablets of aspirin. Their are no adverse affects noted. The heal is progressing well.",
                "Sample 3 - Context Errors": "The patient has been suffering form chest pan for to weeks. She was advice to loose weight and quite smoking.",
                "Sample 4 - Medical Context": "The patience complained off severe hedache and nasea. Blood presure was elevted. Prescribed medication too control hypertention."
            }
            
            selected_sample = st.selectbox("Load sample text:", ["None"] + list(sample_texts.keys()))
            if selected_sample != "None":
                st.session_state.sample_text = sample_texts[selected_sample]
            
            # Text input area (500 characters max)
            default_text = st.session_state.get('sample_text', '')
            user_text = st.text_area(
                "Enter medical text (max 500 characters):",
                value=default_text,
                max_chars=500,
                height=200,
                placeholder="Type or paste medical text here...\nExample: The patient was given too tablets twice daily.",
                key="text_input"
            )
            
            # Check spelling button
            if st.button("🔍 Check Medical Spelling", type="primary"):
                if user_text:
                    st.session_state.checking = True
                    st.session_state.text_to_check = user_text
                else:
                    st.warning("Please enter some text to check.")
        
        with col2:
            st.header("📊 Analysis Results")
            
            if 'checking' in st.session_state and st.session_state.checking:
                text = st.session_state.text_to_check
                words = re.findall(r'\b[a-zA-Z]+\b', text)
                
                # Define full_context here
                full_context = st.session_state.text_to_check
                
                # Find non-word errors
                non_word_errors = []
                for word in words:
                    if word.lower() not in st.session_state.corrector.vocabulary:
                        # Use get_candidates_advanced with context
                        candidates = st.session_state.corrector.get_candidates_advanced(word, context=full_context)
                        non_word_errors.append((word, candidates))
                
                # Find real-word errors (includes homophones now)
                real_word_errors = st.session_state.corrector.detect_real_word_errors_enhanced(text)
                
                # Display errors
                if non_word_errors or real_word_errors:
                    st.subheader("❌ Medical Spelling Errors Found:")
                    
                    # Non-word errors
                    if non_word_errors:
                        st.write("**Non-word errors (not in medical dictionary):**")
                        for error_word, advanced_candidates in non_word_errors:
                            with st.expander(f"🔴 '{error_word}' - Not in medical dictionary"):
                                if advanced_candidates:
                                    st.write("**Medical term suggestions with detailed metrics:**")
                                    
                                    # Create a DataFrame for better visualization
                                    suggestions_data = []
                                    for suggestion, metrics in advanced_candidates[:5]:
                                        suggestions_data.append({
                                            'Suggestion': suggestion,
                                            'Edit Dist': metrics['edit_distance'],
                                            'Levenshtein': f"{metrics['levenshtein']:.2f}",
                                            'Jaccard': f"{metrics['jaccard']:.2f}",
                                            'N-gram': f"{metrics['ngram']:.2f}",
                                            'Cosine': f"{metrics['cosine']:.2f}",
                                            'Frequency': f"{metrics['frequency']*1000:.3f}",
                                            'Bigram': f"{metrics['bigram_prob']*100:.2f}%",
                                            'Trigram': f"{metrics.get('trigram_prob', 0)*100:.2f}%",
                                            'Collocation': f"{metrics.get('collocation_score', 0):.2f}",
                                            'Combined': f"{metrics['combined_score']:.3f}"
                                        })
                                    
                                    df_suggestions = pd.DataFrame(suggestions_data)
                                    st.dataframe(df_suggestions, use_container_width=True)
                                    
                                    # Show the best suggestion prominently
                                    best_suggestion = advanced_candidates[0][0]
                                    best_score = advanced_candidates[0][1]['combined_score']
                                    st.success(f"**Best suggestion: '{best_suggestion}' (confidence: {best_score:.2%})**")
                                    
                                    # Explain the metrics directly (not in nested expander)
                                    st.write("**📊 Metric Explanations:**")
                                    st.caption("""
                                    • Edit Distance: Number of character changes needed
                                    • Levenshtein: Normalized edit distance (higher is better)
                                    • Jaccard: Character set overlap (0-1)
                                    • N-gram: Character sequence similarity (phonetic)
                                    • Cosine: Context-based similarity using TF-IDF
                                    • Frequency: How common the word is in medical texts
                                    • Bigram: Probability based on previous word
                                    • Trigram: Probability based on two previous words
                                    • Collocation: How well word fits medical phrase patterns
                                    • Combined: Weighted average of all metrics
                                    """)
                                else:
                                    st.write("No suggestions found")
                    
                    # Real-word errors
                    if real_word_errors:
                        st.write("**Context-based errors (homophones, confusables, collocations):**")
                        for error_word, position, alternatives in real_word_errors:
                            # Determine error type for better messaging
                            error_type = "Context error"
                            if error_word.lower() in st.session_state.corrector.homophones:
                                error_type = "Homophone error"
                            elif error_word.lower() in st.session_state.corrector.confusables:
                                error_type = "Confusable word"
                            
                            with st.expander(f"🟡 '{error_word}' - {error_type} at position {position + 1}"):
                                # Show context around the error
                                words_list = text.split()
                                start_idx = max(0, position - 2)
                                end_idx = min(len(words_list), position + 3)
                                context_snippet = ' '.join(words_list[start_idx:end_idx])
                                st.info(f"Context: ...{context_snippet}...")
                                
                                st.write("**Better alternatives based on context:**")
                                for alt in alternatives:
                                    # Explain why this alternative is better
                                    explanation = ""
                                    if alt in st.session_state.corrector.homophones.get(error_word.lower(), set()):
                                        explanation = " (homophone - sounds similar but different meaning)"
                                    elif alt in st.session_state.corrector.confusables.get(error_word.lower(), set()):
                                        explanation = " (commonly confused word)"
                                    st.write(f"• **{alt}**{explanation}")
                    
                    # Now show the correction sections OUTSIDE of the error details
                    st.markdown("---")  # Separator
                    
                    # Always show corrected text sections when errors are found
                    # Highlighted text - Always show this
                    st.subheader("📝 Highlighted Medical Text:")
                    highlighted_text = text
                    
                    # Track all corrections for proper highlighting
                    error_positions = {}
                    
                    # Collect non-word errors
                    for error_word, _ in non_word_errors:
                        # Find all occurrences of the error word
                        pattern = re.compile(r'\b' + re.escape(error_word) + r'\b', re.IGNORECASE)
                        for match in pattern.finditer(text):
                            error_positions[match.span()] = ('non-word', error_word)
                    
                    # Collect real-word errors with their positions
                    words_in_text = text.split()
                    current_pos = 0
                    for i, word in enumerate(words_in_text):
                        for error_word, position, _ in real_word_errors:
                            if i == position and word.lower() == error_word.lower():
                                start = text.find(word, current_pos)
                                if start != -1:
                                    end = start + len(word)
                                    error_positions[(start, end)] = ('context', word)
                        current_pos = text.find(word, current_pos) + len(word) if word in text[current_pos:] else current_pos
                    
                    # Sort positions for proper replacement
                    sorted_positions = sorted(error_positions.items(), key=lambda x: x[0][0], reverse=True)
                    
                    # Apply highlighting from end to start to maintain positions
                    for (start, end), (error_type, word) in sorted_positions:
                        if error_type == 'non-word':
                            highlighted_text = (highlighted_text[:start] + 
                                              f"**:red[{word}]**" + 
                                              highlighted_text[end:])
                        else:  # context error
                            highlighted_text = (highlighted_text[:start] + 
                                              f"**:orange[{word}]**" + 
                                              highlighted_text[end:])
                    
                    st.markdown(highlighted_text)
                    
                    # Auto-corrected text section - Always show this
                    st.markdown("---")
                    st.subheader("✨ Auto-Corrected Text:")
                    
                    # Add confidence threshold slider
                    confidence_threshold = st.slider(
                        "Correction Confidence Threshold:",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.3,
                        step=0.1,
                        help="Higher values = only very confident corrections"
                    )
                    
                    # Generate corrected text
                    corrected_text, corrections_made = st.session_state.corrector.generate_corrected_text(
                        text, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Display corrected text in a nice format
                    st.success(corrected_text)
                    
                    # NEW: Show possible corrections section - Always show this
                    st.markdown("---")
                    st.subheader("🔄 Possible Correct Sentences")
                    
                    # Generate multiple possible corrections
                    possible_corrections = st.session_state.corrector.generate_possible_corrections(text, num_variations=3)
                    
                    if possible_corrections:
                        # Display each possibility
                        for idx, possibility in enumerate(possible_corrections, 1):
                            with st.expander(f"Option {idx}: {possibility['strategy']} Strategy ({possibility['confidence']:.0%} confidence)", expanded=(idx==1)):
                                # Show the corrected text prominently
                                st.success(possibility['corrected_text'])
                                
                                # Show metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Confidence", f"{possibility['confidence']:.0%}")
                                with col2:
                                    st.metric("Changes Made", possibility['num_changes'])
                                with col3:
                                    st.metric("Strategy", possibility['strategy'])
                                
                                # Show explanation
                                st.write("**Explanation:**")
                                st.write(possibility['explanation'])
                                
                                # Show detailed corrections if any
                                if possibility['corrections']:
                                    st.write("**Detailed Changes:**")
                                    for correction in possibility['corrections']:
                                        if correction['type'] == 'spelling':
                                            st.write(f"• **Spelling:** '{correction['original']}' → '{correction['correction']}' ({correction.get('confidence', 0.5):.0%})")
                                        elif correction['type'] == 'homophone':
                                            st.write(f"• **Homophone:** '{correction['original']}' → '{correction['correction']}' (sound-alike word)")
                                        elif correction['type'] == 'confusable':
                                            st.write(f"• **Confusable:** '{correction['original']}' → '{correction['correction']}' (commonly confused)")
                                        else:
                                            st.write(f"• **Context:** '{correction['original']}' → '{correction['correction']}' (better fit)")
                                
                                # Add copy button for this option
                                if st.button(f"📋 Use Option {idx}", key=f"use_option_{idx}"):
                                    st.session_state.selected_correction = possibility['corrected_text']
                                    st.success(f"Option {idx} selected! Text copied to clipboard area below.")
                    
                    # Show selected correction if any
                    if 'selected_correction' in st.session_state:
                        st.markdown("---")
                        st.subheader("📋 Selected Correction")
                        st.code(st.session_state.selected_correction, language=None)
                        st.info("Copy the text above to use it")
                    
                    # Legend
                    st.caption("🔴 Red: Non-medical word errors | 🟡 Orange: Context/Homophone/Confusable errors")
                
                else:
                    st.success("✅ No spelling errors detected!")
    else:
        st.info("👆 Please load the medical corpus first using the button in the sidebar.")
        
if __name__ == "__main__":
    create_gui()        