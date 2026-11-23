"""Build the two fixed lists of the names format (run once; the results are part of the format).

names.txt         1024 first names: each name in a text carries 10 bits.
secret_words.txt  4096 everyday words: a word of the secret costs 13 bits instead of 4-5 a letter.

usage: build_lists.py OUT_DIR BIP39_FILE EFF_FILE

BIP39_FILE  https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt
EFF_FILE    https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt  (CC BY 3.0 US)

Both lists must stay as they are: running this again with other inputs would
change the format. It is kept to show where the lists came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAMES = """
James Robert John Michael David William Richard Joseph Thomas Charles Christopher Daniel Matthew
Anthony Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Edward Ronald Timothy Jason
Jeffrey Ryan Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel
Gregory Alexander Patrick Raymond Dennis Jerry Aaron Henry Douglas Peter Adam Nathan Zachary Walter
Kyle Harold Carl Jeremy Keith Roger Gerald Ethan Arthur Terry Sean Austin Noah Lawrence Jesse Bryan
Albert Dylan Bruce Gabriel Alan Juan Logan Wayne Ralph Eugene Randy Vincent Russell Louis Philip
Bradley Liam Oliver Elijah Lucas Owen Caleb Isaac Luke Connor Evan Adrian Colin Simon Oscar Felix
Hugo Theo Finn Jasper Milo Silas Ezra Levi Nolan Declan Callum Angus Fergus Duncan Malcolm Graham
Gordon Stuart Trevor Nigel Clive Derek Neil Rory Niall Cormac Eamon Seamus Brendan Desmond Edgar
Edmund Edwin Ernest Frederick Geoffrey Gilbert Howard Hubert Leonard Lionel Maurice Norman Percy
Reginald Rupert Sidney Stanley Vernon Wilfred Winston Alfred Archie Arnold Bernard Cecil Clarence
Clifford Cyril Dominic Elliot Francis Harvey Julian Marcus Martin Nathaniel Sebastian Spencer
Theodore Tobias Tristan Wesley Xavier
Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret
Sandra Ashley Kimberly Emily Donna Michelle Amanda Melissa Deborah Stephanie Rebecca Sharon Laura
Cynthia Kathleen Angela Shirley Anna Brenda Pamela Emma Nicole Helen Samantha Katherine Christine
Debra Rachel Carolyn Janet Catherine Maria Diane Ruth Julie Olivia Joyce Victoria Kelly Lauren
Christina Joan Evelyn Judith Megan Andrea Cheryl Hannah Jacqueline Martha Gloria Teresa Sara Madison
Frances Kathryn Janice Abigail Alice Julia Judy Sophia Denise Marilyn Beverly Isabella Theresa Diana
Natalie Brittany Charlotte Marie Kayla Alexis Lori Amelia Chloe Claire Eleanor Nora Lucy Stella
Audrey Naomi Lydia Clara Vivian Sophie Josephine Molly Esther Fiona Phoebe Daphne Imogen Gemma
Harriet Matilda Florence Beatrice Agnes Edith Mabel Winifred Dorothy Doris Ethel Gladys Irene
Mildred Muriel Phyllis Sylvia Vera Wendy Yvonne Bridget Maeve Niamh Siobhan Aoife Ciara Orla Sinead
Eileen Maureen Colleen Shannon Erin Moira Isla Elsie Cora Thea Willa Greta
Jose Luis Carlos Jorge Miguel Pedro Manuel Francisco Antonio Alejandro Diego Javier Rafael Fernando
Sergio Pablo Ricardo Eduardo Andres Roberto Raul Enrique Hector Ruben Mario Alberto Arturo Cesar
Emilio Felipe Gonzalo Guillermo Ignacio Joaquin Julio Lorenzo Marcos Mateo Nicolas Ramon Rodrigo
Santiago Tomas Vicente Alvaro Esteban Gustavo Hernan Leonardo Mauricio Octavio Salvador Thiago Bruno
Caio Heitor Joao Matheus Renato Vitor Fabio Marcelo Paulo Rogerio Tiago
Carmen Isabel Marta Lucia Elena Sofia Paula Cristina Pilar Dolores Mercedes Josefina Beatriz Alicia
Silvia Raquel Monica Veronica Gabriela Daniela Valentina Camila Mariana Fernanda Natalia Adriana
Alejandra Catalina Claudia Esperanza Guadalupe Ines Jimena Lourdes Luciana Marisol Paloma Rocio
Ximena Yolanda Bianca Larissa Leticia Juliana Renata Tatiana Vitoria Thais Luiza Helena Manuela
Isadora
Marco Luca Matteo Alessandro Francesco Giovanni Giuseppe Paolo Stefano Riccardo Davide Federico
Giorgio Enrico Pietro Salvatore Vincenzo Luigi Massimo Claudio Fabrizio Gianni Carlo Dario Emanuele
Filippo Gabriele Tommaso Edoardo Giulio Aldo Cesare Ettore Fausto Guido Livio Mauro Nino Orlando
Piero Renzo Sandro Silvio Tullio Valerio
Giulia Chiara Francesca Federica Martina Alessia Elisa Ilaria Eleonora Giorgia Paola Roberta Simona
Caterina Giovanna Antonella Carla Donatella Emilia Fiorella Gabriella Ginevra Lidia Loredana Michela
Nicoletta Ornella Raffaella Serena Stefania Camilla Cecilia Ludovica Matilde Noemi Vittoria Aurora
Gaia Arianna
Pierre Michel Andre Philippe Rene Alain Jacques Marcel Claude Henri Georges Francois Laurent Olivier
Thierry Pascal Julien Guillaume Mathieu Antoine Maxime Romain Quentin Florian Baptiste Clement
Damien Etienne Fabien Gaspard Gilles Jerome Lucien Marius Raphael Sylvain Valentin Yves Yannick
Benoit Cedric Didier Emile Fabrice Gerard Herve Loic Mathis Regis Serge Thibault
Jeanne Francoise Monique Nathalie Isabelle Sylvie Martine Camille Manon Louise Juliette Margaux
Clemence Elodie Amelie Aurelie Celine Colette Delphine Eloise Estelle Gabrielle Genevieve Helene
Laure Lucie Madeleine Marion Mathilde Odette Pauline Sabine Simone Solange Suzanne Therese Valerie
Veronique Virginie Yvette Oceane Maelle Anais Apolline Capucine Agathe Adele
Hans Klaus Wolfgang Jurgen Dieter Manfred Helmut Gunther Werner Heinz Karl Rolf Horst Gerhard Bernd
Rainer Detlef Stefan Andreas Markus Matthias Jonas Lukas Leon Maximilian Moritz Niklas Fabian Philipp
Johannes Lars Sven Torsten Volker Axel Bjorn Dirk Erik Georg Holger Ingo Jens Ludwig Norbert Otto
Ralf Rudolf Siegfried Ulrich Wilhelm Arne Emil Anton Oskar Konrad Friedrich Heinrich Benedikt
Clemens Dominik Gregor Henrik Joachim Kilian Linus Malte Pieter Willem Hendrik Joost Maarten Sander
Thijs Wouter Bram Daan Luuk Jeroen Koen Niels Stijn Teun
Ursula Monika Petra Renate Helga Karin Brigitte Ingrid Erika Gisela Susanne Christa Heike Birgit
Angelika Kerstin Silke Anke Antje Astrid Bettina Dagmar Elke Frieda Gerda Hannelore Ilse Jutta
Katja Lena Leonie Johanna Luisa Ella Lotte Marlene Nele Ronja Svenja Tabea Ulrike Wiebke Annelies
Femke Sanne Fenna Lieke Esmee
Anders Johan Nils Olof Gustav Magnus Mikael Fredrik Mats Bengt Leif Elias Viktor Ludvig Isak Albin
Alvar Sixten Stig Rune Knut Olav Harald Haakon Sigurd Trygve Eirik Gunnar Ragnar Einar Bjarne Jesper
Kasper Mads Soren Rasmus Frederik Mikkel Aksel Jukka Matti Pekka Timo Juha Antti Mika Jari Heikki
Eero Ilmari Onni Eino Veikko
Aino Helmi Venla Sanna Tiina Riitta Leena Anneli Kaisa Elina Minna Noora Oona Pinja Saara Siiri
Tuuli Sigrid Freja Maja Elsa Ebba Alma Signe Linnea Tove Ylva Gunilla Birgitta Margareta Ulla Inger
Britt Solveig Ragnhild Hedda Emilie Frida Tuva Sofie Maren Silje Marit Randi Hilde Turid
Ivan Dmitri Sergei Alexei Andrei Mikhail Nikolai Vladimir Yuri Boris Oleg Igor Pavel Maxim Roman
Kirill Artem Denis Evgeni Fyodor Grigori Ilya Konstantin Leonid Matvei Nikita Pyotr Ruslan Stanislav
Timur Vadim Vasili Yaroslav Zakhar Bogdan Taras Mykola Oleksandr Petro Piotr Krzysztof Andrzej
Tomasz Pawel Marcin Michal Jakub Mateusz Kacper Szymon Wojciech Zbigniew Ryszard Jerzy Tadeusz
Henryk Marek Lukasz Grzegorz Dariusz Mariusz Karol Kamil Maciej Bartosz Dawid Filip Jacek Leszek
Miroslav Milan Zoran Dragan Goran Nikola Marko Bojan Dejan Nenad Predrag Slobodan Vladan Zdenek
Vaclav Jiri Petr Ondrej Jaroslav Karel Radek Matej Vojtech Laszlo Istvan Jozsef Ferenc Gabor Zoltan
Attila Tibor Sandor Balazs Bence Levente Mihai Gheorghe Vasile Florin Cristian Radu Dragos Sorin
Costin Ionut Razvan
Olga Irina Svetlana Ekaterina Anastasia Daria Polina Nadezhda Lyudmila Galina Larisa Marina Oksana
Yulia Ksenia Alina Kira Milana Varvara Zoya Yana Raisa Tamara Nina Zinaida Katarzyna Agnieszka
Malgorzata Joanna Magdalena Ewa Krystyna Dorota Beata Aleksandra Karolina Paulina Zuzanna Hanna
Zofia Wiktoria Oliwia Kinga Iwona Jolanta Grazyna Halina Danuta Bozena Jadwiga Milena Jelena Dragana
Snezana Vesna Ivana Mirjana Biljana Jovana Tijana Katarina Tereza Eliska Adela Barbora Veronika
Marketa Zdenka Eszter Zsofia Katalin Ildiko Reka Dora Boglarka Ioana Andreea Mihaela Raluca
Giorgos Dimitris Nikos Yannis Kostas Christos Panagiotis Vasilis Thanasis Michalis Stavros Spyros
Petros Alexandros Antonis Apostolos Evangelos Fotis Ilias Leonidas Manolis Pavlos Sotiris Stelios
Tasos Thodoris Eleni Katerina Vasiliki Angeliki Dimitra Ioanna Konstantina Despina Chrysa Athina
Danae Elpida Fotini Irini Kalliopi Melina Olympia Rania Stavroula Xenia
Mehmet Mustafa Ahmet Huseyin Hasan Ibrahim Ismail Osman Yusuf Murat Omer Ramazan Halil Suleyman
Abdullah Mahmut Recep Salih Fatih Kemal Orhan Emre Burak Deniz Emir Kaan Kerem Mert Onur Serkan
Tolga Umut Volkan Yigit Baris Berk Cenk Doruk Eren Furkan Gokhan Hakan Ilker Koray Levent Metin
Nihat Okan Ozan Polat Selim Tarik Ufuk Yavuz Zafer Kayra Arda Batuhan Bora Caglar Engin Erdem Ferhat
Gurkan Harun Ilhan Kadir Melih Necati Oguz Sinan Taner Tuncay Ugur Yasin Zeki
Fatma Ayse Emine Hatice Zeynep Elif Meryem Hanife Merve Busra Esra Kubra Ozlem Yasemin Hulya Dilek
Sevgi Songul Sibel Gulsen Aysel Leyla Ebru Derya Pinar Selin Irem Defne Azra Nehir Asli Buse Ceren
Damla Gizem Hande Ipek Melis Nazli Pelin Sevda Tugba Yagmur Zehra Aylin Basak Burcu Cansu Duygu Ezgi
Filiz Gamze Hilal Kader Melek Ozge Rabia Seda Tuba Umay Yeliz Zerrin
Mohammed Ahmed Mahmoud Omar Khaled Hassan Hussein Youssef Mostafa Tariq Karim Samir Nabil Walid Ziad
Rami Fadi Bilal Hamza Idris Jamal Kareem Malik Nasser Rashid Saeed Tamer Wael Yasser Zaid Anwar
Bashir Faisal Ghassan Hadi Imran Jalal Khalil Mansour Nizar Qasim Reza Sami Tahir Usman Yahya
Zakaria Amir Arash Babak Dariush Farhad Hamid Javad Kaveh Mehdi Navid Omid Parviz Ramin Saman Vahid
Fatima Aisha Maryam Zainab Khadija Layla Nour Rania Dina Lina Mona Nadia Salma Yasmin Amira Farah
Hala Iman Jamila Karima Lamia Maha Noura Reem Samira Thuraya Wafa Zahra Dalia Ghada Huda Lubna Malak
Nada Rasha Sawsan Shirin Parisa Nasrin Mina Leila Golnar Azadeh Bahar Darya Elham Fereshteh Mahsa
Negar Roya Setareh Ziba
Aarav Vivaan Aditya Vihaan Arjun Reyansh Ayaan Krishna Ishaan Rohan Rahul Amit Vikram Sanjay Anil
Sunil Ravi Suresh Ramesh Mahesh Rajesh Dinesh Ganesh Mukesh Naresh Prakash Deepak Ashok Vijay Ajay
Manoj Vinod Arun Kiran Nikhil Varun Karan Kunal Gaurav Siddharth Abhishek Akash Ankit Pranav Tarun
Yash Kabir Neel Veer Farhan Salman Irfan Asif Zubair
Aadhya Ananya Diya Saanvi Aanya Anika Navya Myra Priya Pooja Neha Anjali Kavita Sunita Anita Rekha
Meena Geeta Lata Asha Usha Nisha Divya Shreya Sneha Swati Riya Tanvi Isha Kavya Meera Radha Sita
Lakshmi Parvati Durga Kajal Komal Jyoti Preeti Rani Shanti Deepika Aishwarya Aditi Bhavna Chitra
Gauri Indira Jaya Kamala Leela Madhuri Nandini Padma Rupa Sarita Trisha Vidya Yamini Ayesha Farida
Nasreen Rukhsana Shabnam Sana Hina
Hiroshi Takashi Kenji Satoshi Kazuo Akira Yuki Haruto Sota Yuto Riku Kaito Daiki Takumi Kenta Shota
Naoki Tatsuya Yusuke Daisuke Kazuki Makoto Masato Noboru Osamu Shinji Taro Jiro Ichiro Hideo Isamu
Minoru Shigeru Tadashi Yoshio Koji Keisuke Hayato Sakura Yuna Akari Hana Yoko Keiko Naoko Tomoko
Yumi Mariko Noriko Haruka Misaki Nanami Ayaka Kana Rina Saki Asuka Chika Kaori Kumiko Michiko
Sachiko Satomi Shiori Yuka Yuriko
Minjun Seojun Jiho Hyun Jisoo Minho Taeyang Sungmin Dongwoo Jaehyun Kyung Seung Youngho Jiwoo
Seoyeon Hayoon Jiyeon Minji Sooyoung Eunji Hyejin Soojin Nari Dasom
Xiaoming Meiling Jiahui Yifan Zihan Haoran Yuxuan Zixuan Xinyi Yuhan Qiang Zhen Xiu Minh Thanh Tuan
Linh Trang Huong Phuong Quang Khanh Binh Thao Somchai Niran Kanya Malee Siti Dewi Putri Budi Agus
Wayan Ketut Rizal Maricel Rosalinda
Kwame Kofi Kwesi Akua Abena Adjoa Efua Chinedu Emeka Ikenna Obinna Uche Chidi Nnamdi Ngozi Adaeze
Chioma Amaka Ifeoma Nneka Oluwaseun Tunde Femi Segun Bola Kunle Folake Yetunde Funke Bukola Temitope
Tendai Tatenda Farai Rudo Chipo Thabo Sipho Themba Bongani Mandla Lerato Naledi Thandi Zanele Nomsa
Palesa Amara Zuri Imani Jabari Kamau Wanjiru Njeri Otieno Achieng Abebe Haile Selam Tigist Makena
Baraka Juma Neema Zawadi Mwangi Sefu
Yosef Moshe Avraham Yaakov Noam Itai Yonatan Ariel Eitan Nadav Asaf Boaz Ehud Gideon Ilan Lior Meir
Oren Ronen Shimon Tomer Yair Rivka Leah Miriam Tamar Shira Maya Yael Talia Adina Ayelet Efrat Hadas
Ilana Keren Liora Orly Shoshana Tova Yaffa Ziva
Zoe Eva Mia Ida Pia Lea Ada Amy Leo Ian Kai Noa Uma Ela Ece Eda Oya Efe Alp Cem Ali Raj Sai Ren Rin
Mei Yui Aoi Eri Ryo Liv Ulf Ole Uwe Bas Rik Sem Tal Uri Dov Ama Ayo Nur Jun Wei
"""

# First names that are also everyday English words: a note could use them by accident.
NOT_NAMES = """
will may mark bill frank guy max ray art rich jack bob rob pat sue don dick chuck chip buck bud
rose hope joy faith grace ivy iris daisy lily violet hazel ruby pearl amber jade olive holly heather
dawn eve april june august summer autumn winter sky river sage raven robin jay earl duke dean grant
lance pierce reed rusty sandy penny carol angel christian norm gene herb basil cliff dale glen heath
lane brook drew wade chase miles victor major prince king royal justice liberty destiny harmony
melody serenity trinity genesis miracle heaven dream journey legend phoenix paris london india
dallas georgia virginia carolina dakota asia china kenya chad jordan israel hunter parker carter
mason cooper taylor tyler archer sawyer fisher porter walker baker marshall bishop sunny misty candy
sherry brandy ginger pepper cherry coral crystal diamond ebony ivory jewel opal scarlet star mercy
patience honor precious princess queen lady baron lord junior buddy rocky lucky happy dolly kitty
bunny honey sugar love blue red dusty can made long hung dung ion per sultan harsh fang gang bin
ping ting tim jan ann roy joe tom harry nick jean paul john tara dora nada mona mina sita rani hana
"""

SECRET_WORDS = """
the a an and or but not no yes so if then than that this these those there here it its it's is are
was were be been being am do does did done have has had will would can could should shall may might
must I I'm I'll I've I'd me my mine we we'll we're us our you you're you'll your yours he him his
she her they them their who what when where why how which at to of in on by for from with without
into onto out up down over under above below behind before after during until till since about
around near next between through across against along inside outside off away back again once
twice too also only just very really still already never always often soon now later today tonight
tomorrow yesterday morning noon afternoon evening night midnight early late time day days week
weeks month months year years hour hours minute minutes second seconds Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July August September October
November December weekend zero one two three four five six seven eight nine ten eleven twelve
twenty thirty forty fifty sixty hundred thousand million first last half all some any none both
each every many much more most less few other another same new old big small good bad best better
right left north south east west front top bottom middle side end start meet meeting met see saw
seen look watch find found get got go going gone come coming came leave leaving stay wait waiting
bring take took give gave send sent call called text message email phone talk tell told say said
ask know knew think thought want need like love miss keep kept put hide hidden open close closed
lock locked buy bought sell sold pay paid cost money cash card bank price free cheap ready done
safe sure okay ok fine please thanks thank sorry hello hi hey bye dear home house flat room door
gate window floor office work job school station airport hotel cafe bar shop store market park
bridge street road corner city town village country car bus train plane flight ticket key keys
password code number name address package box bag letter paper book table bench chair bed kitchen
garden beach lake river sea mountain forest food water coffee tea lunch dinner breakfast drink
friend friends mom dad mother father brother sister son daughter wife husband boss doctor police
man woman boy girl people person someone anyone everyone nobody something anything everything
nothing don't can't won't didn't doesn't isn't aren't wasn't let's that's there's what's he's
she's they're alone together maybe sharp red blue green black white yellow hot cold wet dry happy
sad tired sick well away far close secret plan problem idea news help stop start run walk drive
fly move change try use make made let turn read write wrote hear heard feel felt remember forget
forgot under usual place thing things way part land landed pick picked drop dropped arrive
arrived arriving pm am o'clock here's where's who's you've we've they've couldn't wouldn't
shouldn't haven't hasn't hadn't asap tonight's today's bringing taking waiting leaving
meeting's usual same place spot entrance exit stairs roof basement parking platform
"""


def main() -> None:
    out, bip39_file, eff_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    bip39 = bip39_file.read_text().split()
    eff = [line.split()[1] for line in eff_file.read_text().splitlines() if line.strip()]
    everyday = {word.casefold() for word in [*bip39, *eff, *NOT_NAMES.split()]}

    names: list[str] = []
    seen: set[str] = set()
    for name in NAMES.split():
        key = name.casefold()
        if key in seen or key in everyday or not (3 <= len(name) <= 10) or not name.isascii():
            continue
        seen.add(key)
        names.append(name)
    print(f"{len(names)} usable names")
    if len(names) < 1024:
        raise SystemExit("not enough names")
    # Evenly thinned, so every part of the world in the list above stays in.
    chosen = sorted((names[at * len(names) // 1024] for at in range(1024)), key=str.casefold)

    words: list[str] = []
    taken: set[str] = set()
    for word in [*SECRET_WORDS.split(), *bip39, *(w for w in eff if w.isalpha())]:
        if word not in taken and len(words) < 4096:
            taken.add(word)
            words.append(word)
    print(f"{len(words)} secret words ({len(SECRET_WORDS.split())} written by hand)")
    if len(words) < 4096:
        raise SystemExit("not enough words")

    out.mkdir(parents=True, exist_ok=True)
    (out / "names.txt").write_text("\n".join(chosen) + "\n")
    (out / "secret_words.txt").write_text("\n".join(words) + "\n")


main()
