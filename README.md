
### nlp4climate – NLP sobre esperança de vida i canvi climàtic 🌍🧠
Aquest projecte analitza textos acadèmics, institucionals i periodístics relacionats amb l’esperança de vida i el canvi climàtic. Utilitza tècniques de NLP com WordClouds, LDA i classificació supervisada per identificar diferències temàtiques i estilístiques segons l’origen del text.

L’aplicació està disponible inicialment en català, i més endavant s’ampliarà a alemany.

🔧 Requisits
Python 3.11

Les llibreries especificades a requirements.txt

Connexió a internet per descarregar el model en_core_web_md de spaCy

📁 Estructura del projecte
bash
Copy
Edit
nlp4climate/
│
├── data/                      # Corpus preprocesats en format .csv i diccionaris .json
├── models/                    # Models entrenats per a classificació i vectorització
├── scripts/
│   ├── main.py                # Script principal de l’app Streamlit
│   ├── ca_content_3.py        # Contingut de la versió catalana de l’app
│   ├── de_content_3.py        # (Opcional) Contingut en alemany
│   └── NLP_functions_3.py     # Funcions auxiliars (visualització, modelatge, etc.)
├── requirements.txt           # Llista de llibreries amb versions
├── setup.sh                   # Script per instal·lar el model spaCy
├── packages.txt               # (per Streamlit Cloud) Paquet addicional necessari
└── README.md                  # Aquest fitxer
▶️ Com executar el projecte
Clona aquest repositori:

bash
Copy
Edit
git clone https://github.com/abenet25/nlp4climate.git
cd nlp4climate
Instal·la les dependències:

bash
Copy
Edit
pip install -r requirements.txt
python -m spacy download en_core_web_md
Executa l’app amb Streamlit:

bash
Copy
Edit
streamlit run scripts/main.py
🚀 Desplegament a Streamlit Community
Per assegurar que el model spaCy es descarrega correctament al núvol:

Inclou un fitxer setup.sh amb:

bash
Copy
Edit
#!/bin/bash
python -m spacy download en_core_web_md
I un fitxer packages.txt amb:

nginx
Copy
Edit
curl
Al desplegar el projecte a Streamlit Community Cloud, indica com a fitxer principal:

bash
Copy
Edit
scripts/main.py
✨ Funcionalitats
Interfície multilingüe (actualment en català)

Núvols de mots per corpus (acadèmic, institucional, periodístic)

Modelatge de temes amb LDA (barres i núvols per tema)

Classificació supervisada de frases segons l’origen:

Acadèmic 🧑‍🎓

Institucional 🏛️

Mediàtic 📰

Comparació de rendiment entre:

Naive Bayes multimonial

Regressió logística

👤 Autora
Ariadna Benet
Projecte desenvolupat com a treball final del Data Science Bootcamp al Data Science Institute.
🔗 https://github.com/abenet25/

📄 Llicència
Aquest projecte està llicenciat sota MIT License.




