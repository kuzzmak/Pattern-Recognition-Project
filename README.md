# ProjektRU

### Detekcija i lokalizacija nenomalnih događaja uporabom metode dubokog učenja

Oblikovati programski sustav za detekciju i lokalizaciju nenomalnih događaja. Sustav neka se temelji na pristupu sličnom
onom koji je opisan u radu: „Spatial-temporal Convolutional Neural Networks for Anomaly Detection and Localization in
Crowded Scenes”, S. Zhou, W. Shen, D. Zeng, M. Fang, Y. Wei, Z. Zhang, (2016). Programski sustav testirati i usporediti
s orginalnim radom na sljedećim bazama: UCSD Dataset, UMN dataset.

### TODO lista

Primarno:

-   [x] (1) izvlačenje SVOI-ja iz slika
-   [x] (2) proučiti konvolucijske mreže i napraviti arhitekturu (u kodu)
-   [x] (3) organizirati oznake dataseta da ih se lako učita u program
    -   [x] (3.1) UCSD ped1
    -   [x] (3.2) UCSD ped2
    -   [x] (3.3) UMN plaza
    -   [x] (3.4) UMN lawn
    -   [x] (3.5) UMN indoor
-   [x] (4) podijeliti onaj drugi dataset u frameove i označiti ih na smislen način
-   [x] (5) pisanje dokumentacije
    -   [x] (5.1) projektni zadatak
        -   [x] (5.1.1) opis projektnog zadatka
        -   [x] (5.1.2) pregled i opis srodnih rješenja
        -   [x] (5.1.3) konceptualno rješenje zadatka
    -   [x] (5.2) postupak rješavanja zadatka
    -   [x] (5.3) ispitivanje rješenja
    -   [x] (5.4) opis programske implementacije rješenja
    -   [x] (5.5) zaključak
    -   [x] (5.6) literatura

Kod:

-   [x] (6) napraviti funkciju za normalizaciju slika
-   [ ] (7) uravnotežiti razrede koji predstavljaju normalne i abnormalne slike u smislu da jednih i drugih ima podjednako

Sekundarno:

-   [ ] (6) napaviti grafičko sučelje
-   [ ] (7) naći ostala dva dataseta
