---
theme: seriph
class: text-center
highlighter: shiki
transition: slide-up
title: CS490 - Predikcija Nekretnina
mdc: true
---

<h1 class="font-extrabold tracking-tight mb-2">Sistem za Predikciju Cena Nekretnina</h1>
<div class="text-gray-400 font-light tracking-wide">Analiza tržišta nekretnina i predviđanje cene objekata</div>

---
layout: default
transition: fade
---

# Životni Ciklus Podataka
<div class="text-gray-400 mt-2 mb-14 text-sm tracking-wide">Arhitektura sistema je dizajnirana da automatski sprovede podatke kroz 5 ključnih faza</div>

<div class="flex justify-between items-center w-full px-2 mt-8">
  <!-- Korak 1 -->
  <div class="flex flex-col items-center w-1/5 relative" v-click="1">
    <div class="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-blue-500/30 flex items-center justify-center text-2xl text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.15)] z-10">
      <div class="i-carbon-cloud-download"></div>
    </div>
    <div class="mt-5 font-bold text-gray-200">1. Prikupljanje</div>
    <div class="text-xs text-gray-500 text-center mt-1">Web Scraping Oglasa</div>
  </div>

  <div class="h-0.5 bg-gray-800 flex-grow mx-2 relative top-[-30px]" v-click="2"></div>

  <!-- Korak 2 -->
  <div class="flex flex-col items-center w-1/5 relative" v-click="2">
    <div class="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-emerald-500/30 flex items-center justify-center text-2xl text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)] z-10">
      <div class="i-carbon-clean"></div>
    </div>
    <div class="mt-5 font-bold text-gray-200">2. Čišćenje</div>
    <div class="text-xs text-gray-500 text-center mt-1">Uklanjanje šuma</div>
  </div>

  <div class="h-0.5 bg-gray-800 flex-grow mx-2 relative top-[-30px]" v-click="3"></div>

  <!-- Korak 3 -->
  <div class="flex flex-col items-center w-1/5 relative" v-click="3">
    <div class="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 flex items-center justify-center text-2xl text-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.15)] z-10">
      <div class="i-carbon-function"></div>
    </div>
    <div class="mt-5 font-bold text-gray-200">3. Inženjering</div>
    <div class="text-xs text-gray-500 text-center mt-1">Kreiranje obeležja</div>
  </div>

  <div class="h-0.5 bg-gray-800 flex-grow mx-2 relative top-[-30px]" v-click="4"></div>

  <!-- Korak 4 -->
  <div class="flex flex-col items-center w-1/5 relative" v-click="4">
    <div class="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-orange-500/30 flex items-center justify-center text-2xl text-orange-400 shadow-[0_0_15px_rgba(249,115,22,0.15)] z-10">
      <div class="i-carbon-machine-learning-model"></div>
    </div>
    <div class="mt-5 font-bold text-gray-200">4. Trening</div>
    <div class="text-xs text-gray-500 text-center mt-1">Učenje šablona</div>
  </div>

  <div class="h-0.5 bg-gray-800 flex-grow mx-2 relative top-[-30px]" v-click="5"></div>

  <!-- Korak 5 -->
  <div class="flex flex-col items-center w-1/5 relative" v-click="5">
    <div class="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-rose-500/30 flex items-center justify-center text-2xl text-rose-400 shadow-[0_0_15px_rgba(244,63,94,0.15)] z-10">
      <div class="i-carbon-chart-line"></div>
    </div>
    <div class="mt-5 font-bold text-gray-200">5. Predikcija</div>
    <div class="text-xs text-gray-500 text-center mt-1">Procenjivanje cene</div>
  </div>
</div>

---
layout: default
transition: slide-left
---

# Faza 1: Prikupljanje Podataka
<div class="text-gray-400 mt-2 text-sm tracking-wide">Automatizovana ekstrakcija, čitanje HTML elemenata i formiranje baze</div>
<div class="flex flex-row">
  <div class="flex flex-col gap-2">
    <div class="flex items-start gap-2 p-3 rounded-lg hover:bg-gray-800/30 transition-colors" v-click>
      <div class="mt-1 bg-blue-500/10 border border-blue-500/20 p-2.5 rounded-lg text-blue-400"><div class="i-carbon-bot text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Dinamički Web Scraper</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Sistem automatski obilazi portal HaloOglase, čita HTML strukturu i vadi ključne parametre poput cene, lokacije i kvadrature.</p>
      </div>
    </div>
    <div class="flex items-start gap-2 p-3 rounded-lg hover:bg-gray-800/30 transition-colors" v-click>
      <div class="mt-1 bg-blue-500/10 border border-blue-500/20 p-2.5 rounded-lg text-blue-400"><div class="i-carbon-map text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Više Tipova Nekretnina</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Podržano je prikupljanje stanova, kuća, zemljišta i garaža u 10 najvećih gradova Srbije (do 500 stranica po gradu).</p>
      </div>
    </div>
    <div class="flex items-start gap-2 p-3 rounded-lg hover:bg-gray-800/30 transition-colors" v-click>
      <div class="mt-1 bg-blue-500/10 border border-blue-500/20 p-2.5 rounded-lg text-blue-400"><div class="i-carbon-document-export text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Strukturiranje Baze</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Sirovi tekst se pretvara u jasne tabele, uklanjaju se duplikati i podaci se sigurno čuvaju za dalju obradu.</p>
      </div>
    </div>
  </div>
  <div class="flex flex-col w-5/6 justify-center gap-4" v-click>
    <img src="./showcase/showcase_1.png" class="rounded-xl shadow-2xl border border-gray-700/50 object-cover" />
    <img src="./showcase/showcase_1.1.png" class="rounded-xl shadow-2xl border border-gray-700/50 object-cover" />
  </div>
</div>

---
layout: default
transition: fade
---

# Faza 2: Čišćenje i Priprema
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Sirovi podaci sa interneta su često oštećeni. Sistem filtrira podatke pre učenja.</div>

<div class="flex flex-row gap-6">
  <div class="flex-1 w-1/3 bg-gradient-to-br from-gray-800/60 to-gray-900/80 p-6 rounded-xl border border-gray-700/60 shadow-lg" v-click>
    <div class="flex items-center gap-3 pb-4 border-b border-gray-700/50">
      <div class="i-carbon-rule text-xl text-rose-400"></div>
      <h5 class="text-lg font-bold text-gray-200 m-0">Logička Validacija Podataka</h5>
    </div>
    <br>
    <ul class="space-y-3 text-gray-400 text-xs leading-relaxed">
      <li class="flex items-center gap-3"><div class="i-carbon-close-outline text-rose-500 text-base"></div> Uklanjanje nepostojećih cena (≤ 0 €)</li>
      <li class="flex items-center gap-3"><div class="i-carbon-close-outline text-rose-500 text-base"></div> Uklanjanje nemogućih kvadratura (≤ 0 m²)</li>
      <li class="flex items-center gap-3"><div class="i-carbon-close-outline text-rose-500 text-base"></div> Brisanje oglasa bez ključnih informacija</li>
    </ul>
  </div>

  <div class="flex-1 w-1/3 bg-gradient-to-br from-gray-800/60 to-gray-900/80 p-6 rounded-xl border border-gray-700/60 shadow-lg" v-click>
    <div class="flex items-center gap-3 pb-4 border-b border-gray-700/50">
      <div class="i-carbon-data-format text-xl text-emerald-400"></div>
      <h5 class="text-lg font-bold text-gray-200 m-0">Tretman Autlajera (IQR)</h5>
    </div>
    <p class="text-xs text-gray-400 leading-relaxed mb-5">
      Sistem računa Interkvartilni opseg (10. do 90. percentila) kako bi prepoznao ekstremno skupe ili jeftine nekretnine koje odstupaju od realnosti.
    </p>
    <div class="bg-emerald-900/20 text-emerald-400 p-2.5 rounded text-[11px] font-mono text-center border border-emerald-500/20">
      Krajnji rezultat: Zadržano ~70% najkvalitetnijih podataka
    </div>
  </div>

  <div class="flex-1 w-1/3 bg-gradient-to-br from-gray-800/60 to-gray-900/80 p-6 rounded-xl border border-gray-700/60 shadow-lg" v-click>
    <div class="flex items-center gap-3 pb-4 border-b border-gray-700/50">
      <div class="i-carbon-magic-wand text-xl text-cyan-400"></div>
      <h5 class="text-lg font-bold text-gray-200 m-0">Standardizacija Podataka</h5>
    </div>
    <ul class="space-y-3 text-gray-400 text-xs leading-relaxed">
        <br>
        <span>Kategorijalne vrednosti se prevode u numerički oblik tako da sistem može da ih obradi bez nametanja pogrešnog redosleda između njih.</span>
        <br>
        <br>
        <span>Numeričke vrednosti se usklađuju na zajedničku skalu kako bi sve informacije imale sličan uticaj tokom obrade.</span>
    </ul>
  </div>
</div>

---
layout: default
transition: slide-up
---

# Faza 3: Inženjering Obeležja
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Kreiranje novih signala iz postojećih podataka kako bi sistem razumeo šta diktira cenu</div>

<div class="flex flex-row gap-6 mt-8 justify-between">
  <div class="flex-1 w-1/3 bg-gradient-to-b from-gray-800/50 to-gray-900/50 p-6 rounded-xl border-t-[3px] border-purple-500 shadow-xl hover:-translate-y-1 transition-transform" v-click>
    <div class="i-carbon-layers text-3xl text-purple-400 mb-5"></div>
    <h4 class="font-bold mb-3 text-gray-200 text-lg">Indeksi Spratnosti</h4>
    <p class="text-sm text-gray-400 leading-relaxed">Algoritam računa relativnu poziciju stana (npr. 3. sprat od ukupno 5) i detektuje kritične tačke: prizemlje ili potkrovlje.</p>
  </div>
  
  <div class="flex-1 w-1/3 bg-gradient-to-b from-gray-800/50 to-gray-900/50 p-6 rounded-xl border-t-[3px] border-amber-500 shadow-xl hover:-translate-y-1 transition-transform" v-click>
    <div class="i-carbon-chart-bubble text-3xl text-amber-400 mb-5"></div>
    <h4 class="font-bold mb-3 text-gray-200 text-lg">Prostorna Gustina</h4>
    <p class="text-sm text-gray-400 leading-relaxed">Računa se prosečna veličina sobe (m² po sobi) kako bi model napravio jasnu razliku između prostranih i skučenih objekata.</p>
  </div>
  
  <div class="flex-1 w-1/3 bg-gradient-to-b from-gray-800/50 to-gray-900/50 p-6 rounded-xl border-t-[3px] border-cyan-500 shadow-xl hover:-translate-y-1 transition-transform" v-click>
    <div class="i-carbon-function-math text-3xl text-cyan-400 mb-5"></div>
    <h4 class="font-bold mb-3 text-gray-200 text-lg">Logaritmovanje</h4>
    <p class="text-sm text-gray-400 leading-relaxed">Cene rastu eksponencijalno. Primenom logaritamskih funkcija, podaci se ravnaju za lakše i preciznije treniranje.</p>
  </div>
</div>

---
layout: default
transition: fade
---

# Faza 4: Istraživačka Analiza
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Vizuelni pregled tržišta pre pokretanja procesorsko zahtevnih AI operacija</div>

<div class="flex flex-row gap-8 items-center">
  <div class="flex-1 w-1/2 space-y-6 mb-2">
    <div class="bg-gray-800/40 p-2 rounded-xl border border-gray-700/50 shadow-md" v-click>
      <h4 class="font-bold text-gray-200 flex items-center gap-3"><div class="i-carbon-chart-histogram text-blue-400 text-xl"></div> Distribucija cena</h4>
      <p class="text-sm text-gray-400 mt-3 leading-relaxed">Histogram pokazuje gde je glavnica ponude. Jasno se vidi pozitivan efekat našeg čišćenja podataka kroz formiranu zvonastu krivu.</p>
    </div>
    <div class="bg-gray-800/40 p-2 rounded-xl border border-gray-700/50 shadow-md" v-click>
      <h4 class="font-bold text-gray-200 flex items-center gap-3"><div class="i-carbon-network-4 text-rose-400 text-xl"></div> Toplotne Mape</h4>
      <p class="text-sm text-gray-400 mt-3 leading-relaxed">Pomažu nam da vidimo korelaciju. Na primer: kvadratura snažno diktira cenu, dok je broj slika na oglasu manje uticajan faktor.</p>
    </div>
  </div>
  <div class="flex-1 w-1/2 flex flex-col gap-4 mx-4" v-click>
    <img src="./showcase/showcase_2.png" class="rounded-xl shadow-lg border border-gray-700/50 h-[160px] w-full object-cover" />
    <img src="./showcase/showcase_3.png" class="rounded-xl shadow-lg border border-gray-700/50 h-[160px] w-full object-cover" />
  </div>
</div>

---
layout: default
transition: fade
---

# Faza 4: Istraživačka Analiza
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Vizuelni pregled tržišta pre pokretanja procesorsko zahtevnih AI operacija</div>
<div class="flex justify-center text-center">
<img src="./showcase/showcase_4.png" class="rounded-xl shadow-lg border border-gray-700/50 w-4/5 object-cover" />
</div>

---
layout: default
transition: slide-left
---

# Faza 5: Trening Modela
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Napredni pristup paralelnog testiranja više algoritama radi pronalaska šampiona</div>

<div class="flex flex-row gap-10 items-center h-[350px]">
  <div class="flex-1 w-1/2 space-y-2 mt-12">
    <div class="flex items-start gap-4" v-click>
      <div class="mt-1 text-orange-400 bg-orange-400/10 p-2 rounded"><div class="i-carbon-model text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Trka 9 Modela</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Sistem istovremeno testira vrhunske regresione algoritme poput XGBoost, LightGBM i Random Forest arhitektura.</p>
      </div>
    </div>
    <div class="flex items-start gap-4" v-click>
      <div class="mt-1 text-orange-400 bg-orange-400/10 p-2 rounded"><div class="i-carbon-settings text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Tuning Parametara</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Kroz korisnički interfejs možemo kontrolisati dubinu analize (broj stabala, dubinu mreže) za svaki model ponaosob.</p>
      </div>
    </div>
    <div class="flex items-start gap-4" v-click>
      <div class="mt-1 text-orange-400 bg-orange-400/10 p-2 rounded"><div class="i-carbon-trophy text-2xl"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Automatska Selekcija</h4>
        <p class="text-sm text-gray-400 mt-1.5 leading-relaxed">Model sa najvećom preciznošću (R² Skor) automatski se proglašava za pobednika i trajno čuva za buduće predikcije.</p>
      </div>
    </div>
  </div>
  
  <div class="flex-1 w-1/2 flex justify-center" v-click>
    <img src="./showcase/showcase_5.png" class="rounded-xl shadow-2xl border border-gray-700/50 max-h-[300px] object-cover" />
  </div>
</div>

---
layout: default
transition: fade
---

# Faza 6: Predikcija i Procena
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Korisnik unosi parametre svoje nekretnine, a sistem generiše asinhronu tržišnu procenu</div>

<div class="flex flex-row gap-10 items-center h-[380px] mt-4">
  <div class="flex-1 w-1/2 flex flex-col gap-4 mx-4" v-click>
    <img src="./showcase/showcase_6.png" class="rounded-xl shadow-lg border border-gray-700/50 h-[170px] w-full object-cover" />
    <img src="./showcase/showcase_7.png" class="rounded-xl shadow-lg border border-gray-700/50 h-[170px] w-full object-cover" />
  </div>
  <div class="flex-1 w-1/2 space-y-8">
    <div class="bg-gradient-to-r from-blue-900/30 to-transparent p-6 rounded-r-xl border-l-4 border-blue-500" v-click>
      <h4 class="font-bold text-gray-200 flex items-center gap-2"><div class="i-carbon-flash text-blue-400"></div> Reakcija u Realnom Vremenu</h4>
      <p class="text-sm text-gray-400 mt-2 leading-relaxed">Sistem vrši brzu transformaciju unosa, ukršta ga sa pobedničkim modelom i daje procenu.</p>
    </div>
    <div class="bg-gradient-to-r from-rose-900/30 to-transparent p-6 rounded-r-xl border-l-4 border-rose-500" v-click>
      <h4 class="font-bold text-gray-200 flex items-center gap-2"><div class="i-carbon-analytics text-rose-400"></div> Kontekst Tržišta</h4>
      <p class="text-sm text-gray-400 mt-2 leading-relaxed">Aplikacija izračunava i prikazuje na grafikonu da li je nekretnina skuplja ili jeftinija od <b>lokalnog tržišnog proseka</b>.</p>
    </div>
  </div>
</div>

---
layout: default
transition: slide-up
---

# Garancija Kvaliteta (Testing)
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Razvoj vođen testovima (TDD) osigurava stabilnost celokupnog arhitekturnog toka</div>

<div class="flex flex-row gap-12 items-center">
  <div class="flex-1 w-1/3 flex flex-col items-center justify-center" v-click>
    <div class="relative flex items-center justify-center w-40 h-40 rounded-full bg-blue-900/20 border-[4px] border-blue-500/30 shadow-[0_0_30px_rgba(59,130,246,0.2)] mb-6">
      <div class="text-6xl font-extrabold text-blue-400">74</div>
    </div>
    <h3 class="text-xl font-bold text-gray-200 text-center mb-1">Automatizovana Testa</h3>
    <p class="text-sm text-gray-500 text-center">Pytest & Tox okruženje</p>
  </div>

  <div class="w-2/3 space-y-4">
    <div class="bg-gray-800/40 p-2 rounded-xl border-l-4 border-emerald-500 flex items-center gap-5 shadow-md hover:bg-gray-800/60 transition-colors" v-click>
      <div class="bg-emerald-500/10 p-3 rounded-lg"><div class="i-carbon-data-check text-2xl text-emerald-400"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Validacija i Preprocesiranje (44)</h4>
        <p class="text-sm text-gray-400 mt-1">Striktna provera inženjeringa obeležja, čišćenja teksta i tretmana autlajera.</p>
      </div>
    </div>
    <div class="bg-gray-800/40 p-2 rounded-xl border-l-4 border-purple-500 flex items-center gap-5 shadow-md hover:bg-gray-800/60 transition-colors" v-click>
      <div class="bg-purple-500/10 p-3 rounded-lg"><div class="i-carbon-model-builder text-2xl text-purple-400"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Trening i Predikcija (13)</h4>
        <p class="text-sm text-gray-400 mt-1">Testiranje Scikit-Learn pajplajnova, čuvanja modela i stabilnosti predikcija.</p>
      </div>
    </div>
    <div class="bg-gray-800/40 p-2 rounded-xl border-l-4 border-amber-500 flex items-center gap-5 shadow-md hover:bg-gray-800/60 transition-colors" v-click>
      <div class="bg-amber-500/10 p-3 rounded-lg"><div class="i-carbon-category text-2xl text-amber-400"></div></div>
      <div>
        <h4 class="font-bold text-gray-200">Polimorfizam Entiteta (17)</h4>
        <p class="text-sm text-gray-400 mt-1">Validacija rutiranja tipa nekretnine i izolacije specifičnih obeležja (kuće vs stanovi).</p>
      </div>
    </div>
  </div>
</div>

---
layout: default
transition: fade
---

# Plan za Dalji Razvoj
<div class="text-gray-400 mt-2 mb-8 text-sm tracking-wide">Naredni razvojni koraci su fokusirani na industrijsku skalabilnost i nove funkcionalnosti</div>

<div class="flex flex-row gap-6 mt-8 justify-between">
  <div class="flex-1 w-1/3 bg-gray-800/40 p-6 rounded-xl border border-gray-700/50 shadow-lg" v-click>
    <div class="bg-blue-500/10 w-12 h-12 rounded-lg flex items-center justify-center mb-5">
      <div class="i-carbon-data-base text-2xl text-blue-400"></div>
    </div>
    <h4 class="font-bold text-gray-200 mb-4">1. Infrastruktura</h4>
    <ul class="text-sm text-gray-400 space-y-3 list-none p-0">
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-blue-500"></div> Migracija baze na PostgreSQL</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-blue-500"></div> Docker kontejnerizacija</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-blue-500"></div> Puni REST API interfejs</li>
    </ul>
  </div>

  <div class="flex-1 w-1/3 bg-gray-800/40 p-6 rounded-xl border border-gray-700/50 shadow-lg" v-click>
    <div class="bg-purple-500/10 w-12 h-12 rounded-lg flex items-center justify-center mb-5">
      <div class="i-carbon-machine-learning text-2xl text-purple-400"></div>
    </div>
    <h4 class="font-bold text-gray-200 mb-4">2. AI Unapređenja</h4>
    <ul class="text-sm text-gray-400 space-y-3 list-none p-0">
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-purple-500"></div> Odvojeni modeli po tipu</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-purple-500"></div> SHAP analitika (objašnjivost)</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-purple-500"></div> Analiza vremenskih serija</li>
    </ul>
  </div>

  <div class="flex-1 w-1/3 bg-gray-800/40 p-6 rounded-xl border border-gray-700/50 shadow-lg" v-click>
    <div class="bg-emerald-500/10 w-12 h-12 rounded-lg flex items-center justify-center mb-5">
      <div class="i-carbon-user-avatar-filled-alt text-2xl text-emerald-400"></div>
    </div>
    <h4 class="font-bold text-gray-200 mb-4">3. Korisničko Iskustvo</h4>
    <ul class="text-sm text-gray-400 space-y-3 list-none p-0">
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-emerald-500"></div> Interaktivne mape lokacija</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-emerald-500"></div> Webhook alerti za pad cene</li>
      <li class="flex items-center gap-2"><div class="i-carbon-checkmark text-emerald-500"></div> A/B testiranje performansi</li>
    </ul>
  </div>
</div>

---
layout: statement
transition: zoom
---

<div class="text-gray-500 text-xl font-bold tracking-widest uppercase">
<h1>Hvala na pažnji!</h1>
</div>