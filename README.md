
# News Mood
Python script to perform a sentiment analysis of the Twitter activity of various news channels.

In this assignment, I create a Python script to perform a sentiment analysis of the Twitter activity of various news sources and present my findings results.

My final output provides a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: BBC, CBS, CNN, Fox, and New York Times.


The first plot features the following:

A scatter plot of sentiments of the last 100 tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
Each plot point reflects the compound sentiment of a tweet.
Each plot point is sorted by its relative timestamp.

The second plot is a bar plot visualizing the overall sentiments of the last 100 tweets from each organization. I've aggregated the compound sentiments analyzed by VADER.

The final Jupyter notebook shows the following:

Pulls last 100 tweets from each outlet.
Performs a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
Pulls into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
Exports the data in the DataFrame into a CSV file.
Saves PNG images for each plot.

From the sentiment analysis results with the five media tweets on March 27, 2018, we can note that:
 
Overall sentiment polarity is positive for BBC, CBS and Fox tweets and lithely negative for CNN and NY Times, their tweet sentiment is almost neutral.

From the overall media sentiment based on tweets, CBS is most positive at 35% and Fox News is the second most positive with 27%.

The results also show that NY Times is most negative with 6% negative polarity and CNN is second with 1% negative polarity, which the closest to neutral.

```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "QKgn6njuGSOLWK1hgBLFvRSmT"
consumer_secret = "W81c1NrGsF2ZS3EKjqLBFW9pEswdoxjA3Xb6JzV4YK7ytmDln9"
access_token = "115752339-YSjcEfJr2FketAuVWuNJEmNV6XNxanP8KuVA9clZ"
access_token_secret = "SmkUm9quhjuIxEu3NP7cY8YhEqpPEblx6Dj2kzw7nEwkl"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Search Term
target_terms = ["@BBC", "@CBS", "@CNN", "@Fox", "@nytimes"]

# Array to hold sentiment
sentiment_array = []
counter = 1

```


```python
# Loop through each Term
for terms in target_terms:
    public_tweets = api.user_timeline(terms, count=100)
    tweetnumber = 1       
    for tweet in public_tweets:
        print("Tweet %s: %s" % (counter, tweet["text"]))
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = tweetnumber
        sentiment_array.append({"Media Source": terms,
                           "Text":tweet["text"],
                           "Date": tweet["created_at"],
                           "Compound": compound,
                           "Positive": pos,
                           "Neutral": neu,
                           "Negative": neg,
                           "Tweet Count": tweetnumber})
        tweetnumber +=1
        counter +=1

```

    Tweet 1: When mother Marie mysteriously leaves the family home, the repercussions are enormous.
    
    #ComeHome | 9pm | @BBCOne |… https://t.co/S4JgnzQlAx
    Tweet 2: 🇩🇪😂 Even if you don't speak German, this is worth watching until the end! 
    #LiveAtTheApollo https://t.co/J3l8oToRbk
    Tweet 3: 🍜 We've got oodles of noodles with recipes for pad Thai, chow mein, ramen, pho and stir-fries.
    👉… https://t.co/JWfrcIhn2X
    Tweet 4: 😬 What does Facebook know about you? https://t.co/lG4ffeCG09
    Tweet 5: RT @BBCTwo: Happy #WorldTheatreDay! *leaves this here and runs away* @SaraPascoe 🎭😂 https://t.co/3XlEb5gr15
    Tweet 6: RT @BBCScotland: Meet the master of the radiator harp
    
    @DaftLimmy returns to BBC Scotland on Thursday 5 April. https://t.co/G7mjFEJx6k
    Tweet 7: RT @bbcthree: 12-year-old Keilan has very severe curvature of the spine. He needs surgery ASAP but the 5 hour operation could leave him par…
    Tweet 8: RT @bbccomedy: Cathy, master of shade. #Mum https://t.co/VjC2ttkaNF
    Tweet 9: RT @BBCR1: (•_•)
    &lt;)   )╯who
     /    \ 
    
      (•_•)
     \(   (&gt; bit
      /    \
    
     (•_•)
    &lt;)   )╯Beyoncé? 
     /    \ 
    
    https://t.co/7hQd9mcKUY
    Tweet 10: RT @BBCWorld: It's 50 years since the death of Yuri Gagarin, the Russian cosmonaut who was the first man to travel into space 👨‍🚀 https://t…
    Tweet 11: ❤️ The first polar bear cub to be born in the UK for 25 years has been filmed adapting to its surroundings. https://t.co/EsxbH18cN5
    Tweet 12: The ancient Greeks thought a life of ‘constant leisure’ was the highest life you could live. https://t.co/sascCq8iks
    Tweet 13: There's over 50 classic cookery shows being served up on @BBCiPlayer this #EasterWeekend. 
    👉 https://t.co/CKECZMGmx8 https://t.co/j3RfAYQvTI
    Tweet 14: 🎭 @MargotRobbie is planning a new TV series, which will give Shakespeare plays a 'female perspective'.
    👉… https://t.co/pTLXfve0ds
    Tweet 15: 🚀🌎 'Mad' Mike Hughes flew his home-made rocket to try and 'prove' the earth is flat. 
    
    🚨 SPOILER: it's not.  https://t.co/f5oNRItwyq
    Tweet 16: RT @BBCTwo: Now we've got our country back... what actually is it? 🤔
    
    #CunkOnBritain starts Tuesday 3 April, 10pm, @BBCTwo. @missdianemorga…
    Tweet 17: RT @BBCBreakfast: A team of abseilers have been roped in to give Cheddar Gorge in Somerset an extreme spring clean. https://t.co/iYmkiizeRi
    Tweet 18: RT @BBCR1: We can't get enough of @george_ezra's Live Lounge 😍
    
    Watch him perform 'Paradise' and cover 'These Days' right here 👉https://t.c…
    Tweet 19: RT @bbcgetinspired: Check out @JesseLingard and some of his @PremierLeague mates showing us their super moves. 🕺🏻 
    
    Show us yours with #sup…
    Tweet 20: When this woman visited an uninhabited Caribbean island, the last thing she expected was to find it covered in plas… https://t.co/cpKW9tl61v
    Tweet 21: About 700,000 people in the UK are on the autism spectrum, with five times as many males as females being diagnosed… https://t.co/PXJMuG76Tn
    Tweet 22: The number of children living in relative poverty in the UK has risen to 4.1m. 
    
    This is the story of Tyler, one of… https://t.co/xFtEAjBTXC
    Tweet 23: Filmed in Nottingham during the worst winter for the NHS on record, groundbreaking series #Hospital returns tonight… https://t.co/PVSIcvCGc9
    Tweet 24: Tonight, George Alagiah explores the fascinating history of Queen Elizabeth II and her beloved Commonwealth. 
    
    The… https://t.co/xXer4FRyTq
    Tweet 25: 🐶😂 It was a Dachsh(und) to the finish line. https://t.co/7VvkU5GfUg
    Tweet 26: ✈️ Joy and Mary flew WW2 planes before any navigation system was installed.
    #RAFat100 https://t.co/yxNMblfTEf
    Tweet 27: Tom Cruise stars in a remake of the 1953 adaptation of HG Wells’s classic novel. 🍿
    
    War of the Worlds | @BBCiPlayer… https://t.co/h19JvTlFSA
    Tweet 28: In her role as head of the Commonwealth, the Queen embarked on her first tour of the nations in 1953.
    
    But by 1970… https://t.co/Av61bahNZF
    Tweet 29: Never let a squirrel nibble your nobble... 🎈🐿😱
    https://t.co/cHm9m6At3m
    Tweet 30: Apple wants to introduce new emojis to better represent people with disabilities. ❤️️ https://t.co/CcJfojmtTa https://t.co/ZRdEhVPMEm
    Tweet 31: A crack that opened up in Kenya’s Rift Valley, damaging a section of the Narok-Nairobi highway, is still growing... https://t.co/T5YocDauYj
    Tweet 32: From hot cross bun gin to Creme Egg Yorkshire pudding, this year’s Easter treats are bigger, weirder and more decad… https://t.co/HvDJmgcB5q
    Tweet 33: RT @bbcpress: Sue Perkins will return to host the 2018 #BAFTA TV awards on Sunday 13 May on @BBCOne. https://t.co/pieSWNAGbH
    Tweet 34: 😂 @BillBailey is NOT a fan of taramasalata. #Room101 
    
    https://t.co/soAAz27c0C
    Tweet 35: The kiwi bird's egg is so large, it takes almost ten days to lay! 🐦🥚😳  #DavidAttenboroughsNaturalCuriosities
    https://t.co/AkZqK9vzZZ
    Tweet 36: 🤔Why do so many celebrities decide to enter politics?
    
    Some have done so more successfully than others...… https://t.co/MkEDpSeLMJ
    Tweet 37: Wishing you could get out of bed just that little bit earlier? 😴
    
    If you need some motivation to set that alarm, he… https://t.co/zsisa2oA4s
    Tweet 38: Could this be an answer to global water shortages? 🏜💧 This machine creates water out of thin air. 
    
    https://t.co/caz4nXMJg5
    Tweet 39: Tonight, @regyates meets people whose lives have been devastated by the Grenfell fire. 
    
    Reggie Yates: Searching fo… https://t.co/HPgtcZuHte
    Tweet 40: Tonight, @mcgregor_ewan and @McgColin celebrate the centenary of the Royal Air Force. 
    
    RAF at 100 with Ewan and Co… https://t.co/nF2iwBP51b
    Tweet 41: The first ever statue of David Bowie has been unveiled in the town where he debuted Ziggy Stardust. ⚡️… https://t.co/lFgROYVkv1
    Tweet 42: When you're enjoying being single and people just can't deal with it. 🙄😂 @kathbum #LiveAtTheApollo 
    
    https://t.co/byHMHWyhPq
    Tweet 43: 🇺🇸🏝🇬🇧 Welcome to Tangier Island, the tiny US island where people speak with a British accent.… https://t.co/1RoM285gRJ
    Tweet 44: 💬 We could listen to him speak all day. 
    
    📽 Sir David Attenborough's voice was just as iconic in the 60s as it is t… https://t.co/zYi3oK5C13
    Tweet 45: Predictions suggest a build-up of about 80,000 tonnes of plastic in the Pacific Ocean is growing rapidly. ♻️🌊💔… https://t.co/uKD9BQWmUi
    Tweet 46: 👽✨🛸 @prattprattpratt stars as a happy-go-lucky hero who joins forces with an unlikely group of aliens. 
    
    Guardians… https://t.co/B9W7kRZ1Qz
    Tweet 47: Weighing just 100g, a newborn panda is one 900th the size of its mum! 🐼💕 #DavidAttenboroughsNaturalCuriosities https://t.co/nKLQh03DJs
    Tweet 48: 🐟Meet SoFi - the soft robot fish developed by MIT to swim among real fish in coral reefs and around the ocean to he… https://t.co/HDa7q7WsHf
    Tweet 49: Ever wondered what made you feel moody? It might be your gut.
    
    👉 https://t.co/l8oMv5gPN2 https://t.co/vq2yhtBOAn
    Tweet 50: RT @BBCScotland: Up your brunch game with perfect poached eggs.
    
    via @bbcthesocial https://t.co/1imXNeOZ9d
    Tweet 51: RT @bbcweather: #Winter may not be done with us yet, as colder continental air fights back against milder maritime air across the UK this w…
    Tweet 52: RT @BBCSport: A simply astonishing confession from Australia captain Steve Smith and batsman Cameron Bancroft 😳 https://t.co/YaE7fBZamq
    Tweet 53: A woman who drinks 30 cans a day says her addiction to energy drinks is worse than gambling.… https://t.co/d11ATW0F0f
    Tweet 54: You learn something new every day. Here's how to poo a baby jaguar. 🤷💩#BigCatsAboutTheHouse https://t.co/Eg5rF1414b
    Tweet 55: When the clocks have gone forward but there's no way you're getting out of bed yet. 🙅‍♂️⏰#DaylightSavings https://t.co/cHBPV3ITsQ
    Tweet 56: "I still see people screaming for help."
    
    This Sunday at 9pm on @BBCTwo, @REGYATES meets the people whose lives wer… https://t.co/RS16yXF17H
    Tweet 57: The story of the last decade of Picasso's life, through the words of family and friends. 🎨
    
    Picasso's Last Stand |… https://t.co/RyiuI8KgSk
    Tweet 58: "Beauty is your inside, it's your personality and what shines from beneath." 
    
    25-year-old rapper Paigey Cakey had… https://t.co/CVGr69Y0oO
    Tweet 59: Cambridge have won the men's and women's Boat Races. https://t.co/COTzXYuo4N 🏆 🚣 #BoatRace2018 https://t.co/6SQjF1ToSt
    Tweet 60: 📺😂 @RomeshRanga is NOT a fan of Gogglebox. #LiveAtTheApollo
    https://t.co/FHm9g6Emss
    Tweet 61: When proposals go wrong... 😳💍 #Doodlebugs 
    https://t.co/8Yi6giWO3S
    Tweet 62: Writer Sara Maitland has lived alone in rural Scotland for 20 years. 
    
    Here are seven valuable life lessons we can… https://t.co/2eAqUlnwuA
    Tweet 63: 🤳How an Instagram video lost me my dream job in fashion: https://t.co/72yWgHdc1k https://t.co/fl3DxMcIL5
    Tweet 64: 🚣 @clarebalding is live from the River Thames as @UniofOxford and @Cambridge_Uni  meet for one of the most iconic e… https://t.co/Ls9rtIYov4
    Tweet 65: RT @bbcthree: The barber helping men with dementia relive their younger lives. https://t.co/p8axQVK836
    Tweet 66: Which animals are likely to become extinct in your lifetime? 🦏💔
    https://t.co/9KwlmcNrWV
    Tweet 67: Don't miss the highlights from the Gymnastics World Cup! 🤸🤸‍♀️🇬🇧
    
    World Cup Gymnastics | @BBCOne | 2:05pm |… https://t.co/7mpufyeEMo
    Tweet 68: How much do you know about horses? Take this quiz to see if you are an equestrian expert…🐴
    
    https://t.co/ChGCB9Xi1W https://t.co/P9UgJ99nsx
    Tweet 69: Meet Hester, the 10-year-old visually impaired skier who's hoping for Paralympic gold one day. ⛷🥇https://t.co/6aiUaTc3yS
    Tweet 70: "The enthusiastic viewer should feel he is almost the man on the spot!" 
    
    In 1949, the BBC announced it would telev… https://t.co/WEoPpkrW1V
    Tweet 71: RT @BBCNewsNI: Prince Harry and Meghan Markle are visiting Northern Ireland https://t.co/ODxLd1LIr8 https://t.co/s0zPTXj3Tk
    Tweet 72: RT @bbcf1: Lewis Hamilton took the first pole of the season in spectacular style ⚡️
    
    https://t.co/a7YDOx0FlQ
    Tweet 73: RT @sportrelief: A humongous THANK YOU to everyone who has taken part, fund-raised &amp; watched over the past weeks &amp; tonight. The #SportRelie…
    Tweet 74: RT @sportrelief: The nation has joined together for this year's #SportRelief pulling out all the stops to make their steps count. Well done…
    Tweet 75: 🥑A Cultural history of the avocado: https://t.co/jo1UyWlVw7 https://t.co/7VF6pBpG57
    Tweet 76: Meet Alexandre, one of the only male performers of ‘baladi’ – also known as belly dancing – in the Middle East. 🎶
    
     https://t.co/K12N1Y6PU7
    Tweet 77: RT @BBCOne: .@GaryLineker, @OreOduba, @ThisisDavina &amp; more kick off the biggest ever night of #SportRelief now on @bbcone. @sportrelief 
    Fo…
    Tweet 78: From @taylorswift13 to @Beyonce: these are the secretive musicians who avoid interviews. 🎤🤐 https://t.co/nqOmaVMzoo https://t.co/sbDga7L5cp
    Tweet 79: Tonight, @GaryLineker, @ThisisDavina and @OreOduba kick off the biggest ever night of @sportrelief! ️⚽️✨🚴🎉… https://t.co/tgbpC76heP
    Tweet 80: Your week, as told by @louistheroux. 📆😂
    https://t.co/Ae6bDBpIBW
    Tweet 81: RT @BBCWales: A #DanceForParkinsons session with @ndcwales for @GetCreativeUK in #Cardiff this week 
    
    Find out what’s happening on the fina…
    Tweet 82: Meet Australian maths teacher Eddie Woo, who has won fans worldwide with his high-energy lessons, posted on YouTube… https://t.co/vTfNDZzA3b
    Tweet 83: RT @BBCOne: No @andy_murray, this is not a dream. @GeriHalliwell really IS in your bedroom making you sing Spice Girls songs 🎤
    
    @sportrelie…
    Tweet 84: These beautiful photographs reveal how refugees in Tyneside have turned to the healing powers of gardening. 🌿📸… https://t.co/2gq0t9FnIR
    Tweet 85: How would you react if you woke up and found Michael McIntyre and Peppa Pig in your bedroom? 
    
    Poor @AndyMurray…
    
    S… https://t.co/7ZP0hUYeKr
    Tweet 86: Spring is here! 🌱🌸🥦 Make the most of seasonal ingredients with these delicious soup recipes. https://t.co/fXp7eJXoJc https://t.co/hA1b2bdSrd
    Tweet 87: Want to get creative this weekend? Here's how to make a hooky rug. 
    #MakeCraftBritain 
    https://t.co/uqibyP09s7
    Tweet 88: RT @BBCSport: He was one of English football's first black players and the British Army's first ever black officer to command white troops.…
    Tweet 89: RT @BBCRadio2: 👻 “I thought the script was really frightening and original.” Martin Freeman chats to @achrisevans about his creepy new film…
    Tweet 90: RT @BBCTwo: Our entire weekend plans. 😴👇 #BigCatsAboutTheHouse https://t.co/vl0zUcCdmI
    Tweet 91: RT @sportrelief: It looks like #TeamGryffindor are storming ahead in the #HogwartsLeague in the #SportRelief app. 
    
    #TeamHufflepuff, #TeamS…
    Tweet 92: ☕️🍰Some blueberry muffins sold by cafes and supermarkets contain more than the recommended daily intake of sugar fo… https://t.co/9bh6K8aMH2
    Tweet 93: Friday night is CELEBRITY FIGHT NIGHT! 🥊🥊
    
    Who'll be victorious? Tune in to #SportRelief to find out!
    
    https://t.co/9ZhgNGmxOD
    Tweet 94: Scrolling through headlines, the world can feel like a pretty dark place. So here are nine reasons to be happy. 😊 🎉… https://t.co/KrNoB0qaqC
    Tweet 95: Would you want secret helpers like these to help you out in nerve-wracking situations? #TheSecretHelpers 
    https://t.co/ua2X4vxKR3
    Tweet 96: Meet the first polar bear cub to be born in the UK in 25 years. 😍
    https://t.co/njq1r6eONE
    Tweet 97: RT @BBCOne: All hail queen Kat! 🙌👑 
    
    #EastEnders @BBCEastenders https://t.co/PLmAc53L6t
    Tweet 98: Istanbul's Blue Mosque looks spectacular. @wmarybeard looks at the art, meaning and significance of calligraphy the… https://t.co/Mv3voUkcvc
    Tweet 99: 🖋 The artist reimagining Islamic calligraphy for the 21st Century: https://t.co/wgbVwZLDEa #Civilisations https://t.co/nxW6Sokjdn
    Tweet 100: In 1918, the very first signs of the Spanish Flu pandemic were kept under wraps. So how did the deadly illness get… https://t.co/zuZryXLihL
    Tweet 101: Count on Entertainer Of The Year nominee @LukeBryanOnline to crash the party with an epic performance at the 53rd… https://t.co/27ua60HTqu
    Tweet 102: Join @eltonofficial and some of today's hottest names in music when they take the stage to perform his most memorab… https://t.co/jz0jaShZIj
    Tweet 103: RT @ACMawards: The ACM for New Vocal Group of the Year goes to @MidlandOfficial! And yes, that really was @Reba on the phone! #ACMawards ht…
    Tweet 104: RT @ACMawards: In case you didn’t know, the ACM for New Male Vocalist of the Year goes to @BrettYoungMusic. Check out his reaction when @Re…
    Tweet 105: RT @ACMawards: Over the weekend @Reba called the ACM New Artist of the Year winners to let them know they had won! Let’s just say our New F…
    Tweet 106: Congratulations to the 53rd #ACMawards New Artist winners @Lauren_Alaina, @MidlandOfficial, and @BrettYoungMusic! W… https://t.co/Fr8H4arwGj
    Tweet 107: New start times in East/Central Time Zones #60Minutes 7:35ET/6:35CT #Instinct  8:35ET/7:35CT #NCISLA 9:35ET/8:35CT… https://t.co/8W5hAeLrvs
    Tweet 108: Don’t miss a minute of the action. Stream the Elite Eight® games LIVE today starting at 2PM ET with a FREE trial of… https://t.co/8NwU8HdiHR
    Tweet 109: RT @MomCBS: That's a wrap on the #Mom panel at #PaleyFest! Thanks for following along! https://t.co/we4JgqPt6P
    Tweet 110: RT @MomCBS: A fan just commented that #Mom helped bring him out of a deep depression. 💜💜💜 #PaleyFest
    Tweet 111: RT @MomCBS: "Go out for it anyway. If you're good for the role, you're good for the role." @theJaimePressly's advice for aspiring actors wi…
    Tweet 112: RT @MomCBS: Mom Co-Creator @GemmaRBaker just pointed out her own #Mom in the audience at #PaleyFest! 💜
    Tweet 113: RT @MomCBS: "I'm not someone in recovery who goes to AA, but I have taken so much away from it...to take one day at a time." - @theJaimePre…
    Tweet 114: RT @MomCBS: "You get to appreciate working with such talented people." - @AnnaKFaris #Mom #PaleyFest
    Tweet 115: RT @MomCBS: “I love this job. I love working with these women. I love working in front of the live audience… It’s alive and it’s fun.” - @A…
    Tweet 116: Get on your feet for @Jason_Aldean, @ThomasRhett, @ChrisStapleton, @KeithUrban, and @ChrisYoungMusic, the five nomi… https://t.co/oT5ogjdj4x
    Tweet 117: Get ready for some sweet games! Stream #5 Clemson vs #1 Kansas LIVE at 7PM ET and #11 Syracuse vs #2 Duke LIVE at 9… https://t.co/4WstgrKNnW
    Tweet 118: RT @SEALTeamCBS: In honor of #NationalPuppyDay... 😍 #SEALTeam https://t.co/4mIZPpiRlU
    Tweet 119: RT @HawaiiFive0CBS: Nothing like a man and his dog! 😍🐶 Happy #NationalPuppyDay to Eddie, the best pup on the Five-0 Task Force! #H50 https:…
    Tweet 120: Game on! 16 teams left and the race to the finish continues tonight. Stream #11 Loyola-Chicago vs #7 Nevada LIVE at… https://t.co/W374rmzzoC
    Tweet 121: Save the date! These are season finales you do NOT want to miss. RT if you're excited! https://t.co/UUQoWsPPSh https://t.co/cDl4WmxMtU
    Tweet 122: Congratulations to all of the @CBSDaytime nominees for the #DaytimeEmmys! See the full list of #DaytimeEmmy nominee… https://t.co/ivJVJWvfsf
    Tweet 123: Female Vocalist Of The Year nominee @MarenMorris will show her fans how it’s done when she takes the stage to showc… https://t.co/PITjmAoFT8
    Tweet 124: The legendary @Reba returns to host the 53rd #ACMawards and she’s proving just how comfortable she is behind the mi… https://t.co/XPXcSPRXqC
    Tweet 125: RT @nancyodell: Told my daughter I'd be presenting at @ACMawards again this year. (Woot woot!We both luv country music!)She took this pic o…
    Tweet 126: RT @ladyantebellum: Ecstatic to announce we'll be performing at the #ACMawards in Las Vegas again this year! https://t.co/Qfhs94j6FR
    Tweet 127: Country superstars @kennychesney, @ladyantebellum, @blakeshelton, and @KeithUrban have just been added to the stell… https://t.co/bJ4If7MacP
    Tweet 128: RT @YandR_CBS: Forever evolving, Forever inspiring, Forever Young and Restless. ❤️ Get ready to celebrate 45 years of #YR starting in just…
    Tweet 129: New start times in East/Central Time Zones: #60Minutes 7:37ET/6:37CT #Instinct series premiere 8:37ET/7:37CT… https://t.co/xT3YKqmu2M
    Tweet 130: Spend your Sunday streaming Second Round games LIVE with a FREE trial of CBS All Access! https://t.co/3P85rXLy4b https://t.co/zbWfirD9Ju
    Tweet 131: RT @instinctcbs: TONIGHT, Dr. Dylan Reinhart rewrites the book on abnormal behavior. Don't miss the premiere of #Instinct at 8/7c! https://…
    Tweet 132: If any duo knows how to rock the stage, it's @FLAGALine. The Vocal Duo Of The Year nominee will perform live at the… https://t.co/FknabB8NQp
    Tweet 133: How is your bracket looking after last night? Stream Second Round games LIVE today with a FREE trial of CBS All Acc… https://t.co/25JlIpgwog
    Tweet 134: Where better to spend #StPatricksDay than the place everybody knows your name? It’s just your luck that every singl… https://t.co/Fom5wmdENL
    Tweet 135: Stars @JakeMcDorman and Nik Dodani will join the cast in the upcoming revival of Murphy Brown coming to CBS.… https://t.co/JCAx29lo0i
    Tweet 136: RT @thegoodfight: Go behind the scenes with costume designer @DanLawsonStyle in "Behind The Style," a new weekly video series all about the…
    Tweet 137: The games have just begun! Continue to stream First Round games LIVE today with a FREE trial of CBS All Access:… https://t.co/YTGsJ48zYP
    Tweet 138: RT @TheTalkCBS: You asked, we answered! The fun never ends when the ladies #KeepTalking and answer your fan questions 🗣💬➡️ https://t.co/ie1…
    Tweet 139: RT @instinctcbs: Dr. Dylan Reinhart is lured back into the field from his life of quiet academia when a certain serial killer makes things…
    Tweet 140: Stream First Round games LIVE today starting at 12PM ET with a FREE trial of CBS All Access! https://t.co/3P85rXLy4b https://t.co/vZow3YD8cb
    Tweet 141: RT @CBSSports: It's the most wonderful time of the year. #MarchMadness https://t.co/e4c9qohqSR
    Tweet 142: Give these ladies some love! @Lauren_Alaina, @DBradbery, @carlypearce, and @RaeLynn are nominated for New Female Vo… https://t.co/IVhwURfJ3S
    Tweet 143: RT @ManWithAPlan: Hungry for more #ManWithAPlan bloopers and behind-the-scenes videos featuring cast like @matt_leblanc, @thelizasnyder, @k…
    Tweet 144: Music stars @MileyCyrus, @edsheeran, @ladygaga, and more will honor the legendary @eltonofficial and his hit songs… https://t.co/UzxARCCLnI
    Tweet 145: RT @thegoodfight: The verdict is in. The new season of #TheGoodFight is 🔥🔥🔥! Stream it now on CBS All Access: https://t.co/FkYSNSXlRb https…
    Tweet 146: RT @MadamSecretary: In less than an hour, #MadamSecretary's Keith Carradine will be taking over the @MadamSecretary Twitter page! Tweet alo…
    Tweet 147: RT @DierksBentley: Take and post a photo of the woman in your life who inspires you daily! Use the hashtag #WomanAmenACM in your post for a…
    Tweet 148: RT @MomCBS: If you missed guest star @KChenoweth in the latest episode of #Mom, not to worry! Watch now: https://t.co/RlvXoGOZ0l https://t.…
    Tweet 149: Give a round of applause to @KelseaBallerini, @MirandaLambert, @Reba, @MarenMorris, and @CarrieUnderwood, the five… https://t.co/Ncp1BTXx6N
    Tweet 150: RT @thegoodfight: Smart, sexy, and sophisticated. See what's coming this season on #TheGoodFight. https://t.co/CuKhx2G50P https://t.co/ygTI…
    Tweet 151: RT @BlueBloods_CBS: Even stand-up guys fall down sometimes. #BlueBloods is new tonight at 10/9c! https://t.co/UOlDm22wWW
    Tweet 152: Today and every day we celebrate the women in our lives who empower and inspire us. Share a story about an influent… https://t.co/9rVtqrElvT
    Tweet 153: Take and post a photo of the woman in your life who inspires you daily! Use the hashtag #WomanAmenACM in your post… https://t.co/7ShhvE48zy
    Tweet 154: RT @thegoodfight: Meticulously constructed. Soapy &amp; sexy. Intoxicating, savage television. 🔥 Here's what critics are saying about #TheGoodF…
    Tweet 155: This just in! @Jason_Aldean, @mirandalambert, @LukeBryanOnline, and many more are set to perform at the 53rd Academ… https://t.co/mfxw2VxzU4
    Tweet 156: Meet the ensemble of talented actors slated to join $1, a new mystery series coming to CBS All Access:… https://t.co/QoyYv7vxwg
    Tweet 157: Will @Jason_Aldean, @garthbrooks, @LukeBryanOnline, @ChrisStapleton, or @KeithUrban be named Entertainer Of The Yea… https://t.co/rMD8zjeX3s
    Tweet 158: RT @thegoodfight: It feels good to be back. 👠💄🔥 The season 2 premiere of #TheGoodFight is now streaming, exclusively on CBS All Access: htt…
    Tweet 159: RT @thegoodfight: Tomorrow, #TheGoodFight is back. Stream the season 2 premiere only on CBS All Access: https://t.co/tNFR8LBJO2 https://t.c…
    Tweet 160: Who are the trailblazing women in your life that inspire you? Join CBS and the ANA's #SeeHer initiative, celebratin… https://t.co/M0KqZ41Bes
    Tweet 161: Join @maria_bello, @aishatyler and @TeaLeoni in celebrating the accomplishments of women who have contributed to th… https://t.co/MefESBeFL3
    Tweet 162: In honor of Women's History Month, CBS and the Association of National Advertisers' (ANA) #SeeHer initiative will p… https://t.co/2wtYxKJVuO
    Tweet 163: RT @ZoeListerJones: Tonight’s an all new Life In Pieces and it’s directed by my ride or die @nataliaanderson!!!… https://t.co/2LPfmyLWrY
    Tweet 164: RT @MarenMorris: Hot damn! Woke up from my post-wisdom teeth haze to find out I’m up for 4 @ACMawards ! So honored, especially for the Dear…
    Tweet 165: RT @KelseaBallerini: Ohhhhh goodness. Incredible. Thank you thank you thank you. #female https://t.co/1ZTYjNfQeF
    Tweet 166: RT @KeithUrban: ACMs...... HOLY SMOKES!!!!! MAD LOVE TO U ALL THIS MORNING  FOR THESE INCREDIBLE NOMINATIONS. I’M EXTREMELY GRATEFUL!!!!!!!…
    Tweet 167: RT @ACMawards: Congratulations to this year’s #ACMawards Video of the Year nominees:
    “Black” - @DierksBentley
    “It Ain’t My Fault” - @Brothe…
    Tweet 168: RT @ACMawards: Please give a round of applause to this year’s #ACMawards Entertainer of the Year nominees: @Jason_Aldean, @GarthBrooks, @Lu…
    Tweet 169: .@ChrisStapleton, @ThomasRhett, @mirandalambert and more are all nominated for awards at Country Music's Party of t… https://t.co/Vm1vXRUDYJ
    Tweet 170: The Queen of Country, @Reba, is returning to host the 53rd #ACMawards on Sunday, April 15 at 8/7c. Here are a few o… https://t.co/Iqzz6Gql01
    Tweet 171: RT @survivorcbs: It’s time! #Survivor https://t.co/YPk6cGWrUA
    Tweet 172: RT @CBSThisMorning: TOMORROW: The nominees for the 2018 @ACMawards will be announced live by the one-and-only, @Reba! 
    
    Watch on @CBS in ou…
    Tweet 173: RT @thegoodfight: From the set design and costumes to hair and makeup, the production quality is truly next-level. Take a peek inside the u…
    Tweet 174: RT @LivinBiblically: The fun continues on Facebook! The #LivingBiblically cast is live to talk about tonight’s premiere. Tune in here: http…
    Tweet 175: RT @KevinCanWaitCBS: Can you get all the way through these #KevinCanWait bloopers without laughing?! @KevinJames,@LeahRemini and the rest o…
    Tweet 176: RT @ACMawards: That’s right! @Reba is headed to @CBSThisMorning on Thursday, March 1 to announce this year’s #ACMAwards' nominees. Tune in…
    Tweet 177: RT @ScorpionCBS: You can't hack your way to a 197 IQ, but you are well on your way with these Genius Facts from #TeamScorpion! 💻 You can be…
    Tweet 178: RT @SuperiorDonuts: You can always count on @DavidKoechner for a laugh! Did your favorite Tush moment make the list? Catch a new #SuperiorD…
    Tweet 179: RT @TheTalkCBS: TODAY: We loved them together then &amp; we love seeing them together now! Welcome back to the show @THESaraGilbert​'s good fri…
    Tweet 180: RT @thegoodfight: As foundations begin to crumble, our characters struggle to make sense of this new dystopian world. The cast teases what'…
    Tweet 181: #LivingBiblically's @linzkraft and @jrfergjr appeared on @KCBS's Facebook Live this morning, talking all about what… https://t.co/4RebcHuuMQ
    Tweet 182: RT @CBSSports: Introducing CBS Sports HQ, a New 24/7 Direct-to-Consumer Streaming Network for Sports News, Highlights, &amp; Analysis.
    
    Stream…
    Tweet 183: RT @CBSBigBrother: It’s down to the final 5 celebrity Houseguests, and anyone could take home the grand prize! Tune in NOW to watch the #BB…
    Tweet 184: RT @startrekcbs: Binge the entire first season of #StarTrekDiscovery. All episodes now streaming exclusively on CBS All Access: https://t.c…
    Tweet 185: RT @thegoodfight: #TheGoodFight returns in 1 week. Season 2 premieres Sunday, March 4. https://t.co/nomCao1GWp https://t.co/BOn6bOe9Tb
    Tweet 186: RT @thegoodfight: This is our new favorite thing. Christine Baranski debuted #TheGoodFight the Musical on @colbertlateshow last night! 🎵🎤…
    Tweet 187: RT @LivinBiblically: Confession time: have YOU ever hit the "close door" button in an elevator while somebody was approaching? The cast of…
    Tweet 188: RT @CBSEyeSpeak: Mark your calendars! #CBSEyeSpeak kicks off March 14 with The EYE Speak Summit. Follow our page for more details! https://…
    Tweet 189: RT @CBSEyeSpeak: Proud to announce a new CBS initiative, promoting female empowerment and developing the next generation of leaders through…
    Tweet 190: RT @LivinBiblically: When you're living by the Bible, it's good to have a priest and a rabbi on call (provided they answer their phones, th…
    Tweet 191: RT @thegoodfight: Chicago lawyers are being hunted and the world is going insane. 
    
    The new season of #TheGoodFight premieres Sunday, March…
    Tweet 192: Ready for some larger than life competition? This new series from @MarkBurnettTV will premiere in summer 2018.… https://t.co/gDXHLdIJ5v
    Tweet 193: With tournament dreams on the line, make sure to stream these college basketball matchups on CBS All Access:… https://t.co/SGkYUZrQWB
    Tweet 194: RT @LivinBiblically: While Chip's sticking to the Bible's original rules, the cast of #LivingBiblically has given them a more modern makeov…
    Tweet 195: Casting News! Peter Mark Kendall, Michael Gaston, Greg Wise, Rade Šerbedžija, Zack Pearlman, and Keye Chen join the… https://t.co/GFob2KrD8H
    Tweet 196: RT @BullCBS: The verdict is in...#Bull is the perfect Valentine! ❤️ Happy #ValentinesDay! https://t.co/poEejI4AnC
    Tweet 197: RT @NoActivityCBS: Car 27 reporting: Season 2 of #NoActivity coming soon!
    
    Binge season one now on CBS All Access: https://t.co/yvxoQMeyhN…
    Tweet 198: RT @LivinBiblically: Against all odds (and the advice of his God Squad), Chip is determined to live life by the Good Book. Think you could…
    Tweet 199: RT @thegoodfight: Christine Baranski reflects upon the spectacular metamorphosis of her character in #TheGoodFight's first season. Revisit…
    Tweet 200: RT @startrekcbs: Binge the entire first season of #StarTrekDiscovery. All 15 episodes now streaming on CBS All Access: https://t.co/lKLaptP…
    Tweet 201: England's health agency is warning parents to be aware of the signs and symptoms of scarlet fever as infections con… https://t.co/vKJ6QOKqmv
    Tweet 202: Blue states are far more likely to lose money and power over Census citizenship question | Analysis by CNN's Harry… https://t.co/13EWDxnd6X
    Tweet 203: An Ohio fertility clinic says more than 4,000 eggs and embryos were affected by a freezer malfunction, double the n… https://t.co/0EZ5F2Ua6p
    Tweet 204: Wall Street bonuses soar 17% to an average of $184,200 https://t.co/qlOQGfehL8 https://t.co/1bw7FWYwwl
    Tweet 205: Waymo and Jaguar unveil a self-driving, electric SUV https://t.co/DTylJKCuML https://t.co/2aD3JXwfGM
    Tweet 206: 11 reasons (besides the NCAA Final Four) to visit San Antonio https://t.co/1YGqpmrZKz via @CNNTravel https://t.co/Vi45YehYXR
    Tweet 207: Former Disney Channel star Caroline Sunshine is joining the White House press team https://t.co/rOYDj2tPs1 https://t.co/xamwbkWP7h
    Tweet 208: Walmart says it will stop selling Cosmopolitan magazine in checkout lines 
    
    The National Center on Sexual Exploitat… https://t.co/oJWSfJKGk2
    Tweet 209: Prince had a "exceedingly high" concentration of fentanyl in his body when he died https://t.co/xzXGWvrqky https://t.co/5CBfqJYBNs
    Tweet 210: An Ohio fertility clinic says more than 4,000 eggs and embryos were affected by a freezer malfunction, double the n… https://t.co/2kUalNEtFo
    Tweet 211: President Trump has privately floated the idea of funding construction of a border wall with Mexico through the US… https://t.co/OVJnG1Arkl
    Tweet 212: Sean Penn, who has done work in Haiti for years, says he felt "deep hurt" following the controversy over Pres. Trum… https://t.co/jkTdsZtERk
    Tweet 213: Actor Sean Penn smoked a cigarette during an interview on "The Late Show With Stephen Colbert," and viewers had a s… https://t.co/DYMvSOouIc
    Tweet 214: RT @CNNTonight: The White House defends President Trump's silence on allegations leveled against him by Stormy Daniels https://t.co/xPKWwph…
    Tweet 215: Larry Nassar's former boss at Michigan State University used his power to sexually assault, harass, and solicit nud… https://t.co/X7BiNzGTGV
    Tweet 216: RT @AC360: As his administration doubles down on building a border wall, President Trump suggests the US military may foot the bill https:/…
    Tweet 217: Progressives, states and civil rights advocates are preparing a flurry of legal challenges to the Trump administrat… https://t.co/COBpw3wtor
    Tweet 218: How Russian President Vladimir Putin's arrogance handed UK Prime Minister Theresa May a diplomatic coup… https://t.co/Nub4iViDPH
    Tweet 219: Is bread crust more nutritious than its inner crumb? The variety you choose may matter more than whether you eat th… https://t.co/clki6ZwPYZ
    Tweet 220: North Korean leader Kim Jong Un met with the China's Xi Jinping during a surprise trip to Beijing this week, state… https://t.co/UHIZDKXeob
    Tweet 221: In Trump's world, once you check in, you rarely check out https://t.co/bHZg0WgmkX https://t.co/uWZA9cEJu8
    Tweet 222: The children are Democrats' future | By Jesse Ferguson via @CNNOpinion https://t.co/xBaMfKTzhZ https://t.co/3iFNq6BHav
    Tweet 223: FBI Director Christopher Wray said he is doubling the number of FBI personnel tasked with reviewing a large set of… https://t.co/EJynfU7wlW
    Tweet 224: Attorney for brother of slain DNC staffer Seth Rich: Internet activists are spreading lies "as far and wide" as the… https://t.co/XUbhZLGXtJ
    Tweet 225: Government ethics lawyers advised Ivanka Trump to make sure to keep her White House role separate as she planned to… https://t.co/eeBfMRZNs5
    Tweet 226: Michael Cohen — President Trump's loyal fixer https://t.co/I3QjtsNaAZ https://t.co/vEdKQK669d
    Tweet 227: Is the White House Counsel's Office looking into Jared Kushner? The answer isn't clear. https://t.co/lbQxxFZlPX https://t.co/Fdas6E1nO6
    Tweet 228: RT @AC360: Seth Rich's brother sues right-wing activists, Washington Times over conspiracy theories: @GaryTuchmanCNN reports https://t.co/1…
    Tweet 229: Keeping Them Honest: The President sees himself as a “counter-puncher,” but when it comes to Stormy Daniels, all we… https://t.co/i7sXcnh0mI
    Tweet 230: The US needs to borrow almost $300 billion this week https://t.co/ZYuVjMl9H5 https://t.co/xqpI8iWojN
    Tweet 231: China raised eyebrows this month by announcing it will give the Economic Community of West African States a $31.6 m… https://t.co/dgRkS4pNdV
    Tweet 232: RT @AC360: China confirms: Kim Jong Un made a visit to Beijing https://t.co/mCREpkhOLd https://t.co/qoCsm4ZYzD
    Tweet 233: Former President George W. Bush may have retired his political penny loafers, but his dancing shoes? Those haven't… https://t.co/K4DjLuftKI
    Tweet 234: "Tonight, the President, still uncharacteristically silent." With the news dominated by stories of alleged affairs… https://t.co/ywrY4xrHbh
    Tweet 235: After spending 23 years in prison for a crime he didn't commit, Nevest Coleman is back working as a groundskeeper f… https://t.co/3wnuYj9ZXQ
    Tweet 236: North Atlantic right whales may be on edge of extinction. There's been zero births this year.… https://t.co/wkHa5e2fwD
    Tweet 237: Presidential misspellings create a spike in dictionary searches https://t.co/clvvRb4A5j https://t.co/jChXBtd8qI
    Tweet 238: President Trump has appointed to his cabinet several people who've served as conservative TV pundits, many on Fox N… https://t.co/ENk3znOZj5
    Tweet 239: Former President Barack Obama says he aspires to create "a million young Barack Obamas or Michelle Obamas" who will… https://t.co/l7yBatqodJ
    Tweet 240: The brother of slain Democratic National Committee staffer Seth Rich is suing right-wing activists and the Washingt… https://t.co/R4cs9PSQVQ
    Tweet 241: Cops in Tempe, Arizona, now have AR-15s strapped to the back of their motorcycles https://t.co/olcEquajNB https://t.co/xGVdtixqVC
    Tweet 242: RT @OutFrontCNN: "Tonight, the President, still uncharacteristically silent." - Amid a news cycle dominated by stories of alleged affairs w…
    Tweet 243: Three members of the Australian cricket team will be sent home from Johannesburg after admitting during a post-matc… https://t.co/1UMvm0nLLv
    Tweet 244: Stunning photos capture Egypt's ancient underworld https://t.co/bW90Z7OZ7o https://t.co/8abhSDTKEu
    Tweet 245: JUST IN: North Korea leader Kim Jong Un made a surprise two-day trip to Beijing, Chinese state media has confirmed… https://t.co/IW7YZvzjLR
    Tweet 246: Heineken has pulled an ad with the tagline "Sometimes lighter is better" after critics slammed it as racist.… https://t.co/8LXiHLTqQH
    Tweet 247: A new NYPD internal investigation criticizes the way the department handles sexual assaults https://t.co/UALy5XvuiA https://t.co/TNF0zTBLjm
    Tweet 248: "They didn't have to kill him like that. They didn't have to shoot him that many times." The grandmother of Stephon… https://t.co/WqdgsJQJs6
    Tweet 249: Veterans Affairs Secretary David Shulkin appears to be on thin ice with the White House, but some major veterans gr… https://t.co/Tz47JBpTW1
    Tweet 250: Former President Barack Obama says he aspires to create "a million young Barack Obamas or Michelle Obamas" who will… https://t.co/Dk8k01daZO
    Tweet 251: Former Disney Channel star Caroline Sunshine is joining the White House press team https://t.co/3kRTmq4lYC https://t.co/EyFCp1K3v3
    Tweet 252: "The widespread prevalence of endometriosis — and the lack of any long-term treatment options — is nothing short of… https://t.co/qygfw6aVFD
    Tweet 253: RT @CNNSitRoom: An official with deep knowledge of North Korea told CNN there was a "strong possibility" that North Korean leader Kim Jong…
    Tweet 254: Uber's self-driving permit in California goes until March 31 -- and the company said it will let the permit expire… https://t.co/sJWdSR2uTu
    Tweet 255: RT @CNNOpinion: Learn from the Holocaust and stop the massacre in Syria, write @EvaMozesKor and @MhdAGhanem https://t.co/urdH6TeosD
    Tweet 256: The CEO of Waymo, the driverless car division of Google parent company Alphabet, said his company's driverless cars… https://t.co/BSqPEppylZ
    Tweet 257: Some Aetna customers could see lower drug prices next year https://t.co/lYDhNfdyxU https://t.co/WFnbBn5HhA
    Tweet 258: Don't be an April fool and miss out on what's streaming on Netflix, Hulu and Amazon Prime next month https://t.co/sf2jB37GxA
    Tweet 259: An appeals court said Google violated copyright laws when it used Oracle's open-source Java software to build the A… https://t.co/MEw6qBehlk
    Tweet 260: Researchers have detailed the structure and distribution of spaces in your body that they say represent a newfound… https://t.co/QAaCP21k8T
    Tweet 261: The NFL and Nike just announced that they would be extending their partnership for another eight years https://t.co/bAROWhKLHg
    Tweet 262: Six days after a ransomware attack shut down the City of Atlanta's online systems, officials are telling employees… https://t.co/okyUlZFUM5
    Tweet 263: The census has always been a weapon of political power | Analysis by CNN's Gregory Krieg https://t.co/2eHizX8TE7 https://t.co/Cr6gCfy237
    Tweet 264: Steven Spielberg nixes Carl's Jr.'s "SpielBurgers" dreams https://t.co/iz5DyMTxUD https://t.co/K5Vu9d4yHm
    Tweet 265: A former professional clown is running for Congress in South Carolina https://t.co/GMfVUgWMob https://t.co/pSpDLGTqgX
    Tweet 266: There's not a 'snowflake's chance in hell' of repealing the Second Amendment | Analysis by CNN's Z. Byron Wolf… https://t.co/yzVur4ZfQ0
    Tweet 267: White House press secretary Sarah Sanders defended President Trump's silence on allegations leveled against him by… https://t.co/oycTPyBTn1
    Tweet 268: Secretary of Defense James Mattis on President Trump’s newly named national security adviser John Bolton: “I hope t… https://t.co/WHpD4xHuTI
    Tweet 269: Warren Buffett swooped in to rescue General Electric once. Could he do it again? https://t.co/0aW43nOaAa
    Tweet 270: Why the Trump administration's plan to put a citizenship question back on the US census is a big deal… https://t.co/TD01oqVqva
    Tweet 271: The US general in charge of the nation's nuclear arsenal has issued a stark warning that Russia and China are aggre… https://t.co/wFSuV1etYq
    Tweet 272: Michael Phelps calls on the US Olympic Committee to do more to help athletes struggling with depression… https://t.co/Y9pZFED1vl
    Tweet 273: RT @CNNSitRoom: Most Americans say US President Trump isn't doing enough to cooperate with special counsel Robert Mueller's investigation i…
    Tweet 274: Nearly two thirds of Americans approve of President Trump's plan to meet with North Korean leader Kim Jong Un, acco… https://t.co/6I0Ao6SyVw
    Tweet 275: New CNN/SSRS poll: Should Robert Mueller be able to investigate Trump’s finances?
    - Yes 67%
    - No 28%… https://t.co/Q8waAySeZj
    Tweet 276: Hormone replacement may prevent belly fat in postmenopausal women, according to a new study https://t.co/fyYbOSWWfc
    Tweet 277: Most Americans say President Trump isn't doing enough to cooperate with special counsel Robert Mueller's investigat… https://t.co/IzArFrCj0M
    Tweet 278: JUST IN: Nearly two thirds of Americans approve of President Trump's plan to meet with North Korean leader Kim Jong… https://t.co/E8p8cpdLWD
    Tweet 279: "Never mind that an undereducated population is bad for democracy -- it's good for the GOP, and so Republicans are… https://t.co/eGgtXnaEP5
    Tweet 280: Stormy Daniels' lawyer: Donald Trump isn’t speaking out about the affair allegations because “my client is telling… https://t.co/UHS6v2Mc2c
    Tweet 281: The White House defends President Trump's silence on allegations leveled against him by Stormy Daniels… https://t.co/GGtqrtq5lh
    Tweet 282: The NRA insisted it did not use foreign funds for election-related purposes, even as the group acknowledged it acce… https://t.co/wFcSX2AffF
    Tweet 283: Why dismissing Stormy Daniels' story would be a mistake | By Carol Costello via @CNNOpinion https://t.co/Rw77AFywo2 https://t.co/sPuNqi1TOF
    Tweet 284: RT @TheLeadCNN: Rep. Lieu: I support US move to expel Russian diplomats, but we have a long way to go for the Trump Administration to “[sto…
    Tweet 285: Tesla stock slid 8% following a string of bad news, including the NTSB's announcement that it's investigating a fat… https://t.co/8v3hjShQv4
    Tweet 286: Facebook's stock has plunged 18% amid its data scandal, wiping out nearly $80 billion in market value… https://t.co/Y7l6pDeZwJ
    Tweet 287: RT @TheLeadCNN: Rep. Lieu: People need to have faith that the census data is accurate, “and if you’re an immigrant, you might not want to t…
    Tweet 288: Israeli Prime Minister Benjamin Netanyahu hospitalized for a high fever and cough, his office says… https://t.co/nobdnlqqEi
    Tweet 289: Nvidia is temporarily pausing its self-driving car tests on public roads a week after a car from Uber, which incorp… https://t.co/VzWph7rVGz
    Tweet 290: Secretary Zinke, we can't afford to ignore diversity | By Andre Perry via @CNNOpinion https://t.co/Z0mq8ZggbN https://t.co/npCcdgkRrl
    Tweet 291: US Ambassador to the UN Nikki Haley on Russia: Their “blatantly false narratives will not keep us from telling the… https://t.co/FMnyi5lmL6
    Tweet 292: The Dow drops about 345 points, completely reversing a 244-point gain from early in the day, as tech stocks get cru… https://t.co/dwEBWWjmkA
    Tweet 293: The US Supreme Court prepares for a right turn https://t.co/5XsbK13Nhx https://t.co/RS21gJQV4X
    Tweet 294: Apple co-founder Steve Jobs warned about privacy issues in tech at a conference in 2010. Facebook's Mark Zuckerberg… https://t.co/cMxBuIZItP
    Tweet 295: A group of fair housing organizations have filed a lawsuit against Facebook, alleging that its advertising platform… https://t.co/3MfFvBLey6
    Tweet 296: Washington apples and cherries are caught in the trade fight between the United States and China… https://t.co/9jyta5qzmx
    Tweet 297: RT @CNNMoney: Dow drops 345 points, reversing an earlier rally. Nasdaq plunges 2.9% on tech sell-off. Nvidia, Facebook tumble. https://t.co…
    Tweet 298: Not enough was done to stop Russian trolls, former CIA Director Michael Hayden says https://t.co/IWNEqtVw7A https://t.co/PF0CiO23j4
    Tweet 299: RT @CNNMoney: Dow drops 450 points as tech stocks get crushed. The selloff follows Monday's 690-point spike. https://t.co/4FuWC7XE3O https:…
    Tweet 300: Watch former President George W. Bush bust a move on the dance floor https://t.co/QNH2dfEQOT https://t.co/8JWDbQdLQi
    Tweet 301: I’ve been watching Big Little Lies lately, and the portrayal of the complexity of abusive relationships is both exc… https://t.co/qMNxveax5O
    Tweet 302: @RedRoxProjects Thank you! 💐
    Tweet 303: @OphelieLechat 💓
    Tweet 304: This, a thousand times for job interviews or ”I just want to catch up and talk about things”. https://t.co/5AA2d4vkwd
    Tweet 305: @SaskiaFairfull North Melbourne Meat Market.
    Tweet 306: @SaskiaFairfull Thank you!
    Tweet 307: This incredible venue plus my art direction skills. I’m pretty proud of myself.
    
    💖 https://t.co/st8XZOoc4y
    Tweet 308: Have you ever seen a conference releasing a diversity report, even if it’s only in the spectrum of gender?
    
    I haven… https://t.co/e819on1BUh
    Tweet 309: @darngooood I’m perpetually inspired by your clothing style. ✨😭😍
    Tweet 310: @mspowahs ugh damn Facebook, the root of evil
    Tweet 311: Company: Diversity and inclusion are really important to us!
    
    Company has:
    
    → Zero women
    → Women only in support, m… https://t.co/MZmNK67D6a
    Tweet 312: RT @zeynep: Show me your budget and business model, and I’ll know your actual priorities. The rest is mostly PR. https://t.co/hy3XyoW3Lt
    Tweet 313: @jennwrites the voices are coming from inside the house
    Tweet 314: @lara_hogan Congratulations! You both look fantastic. 💝
    Tweet 315: RT @menajew: There’s a common misconception that you can’t be Disabled and happy. That you can’t express joy, and if you do, you must not b…
    Tweet 316: Meanwhile in Poland, where the government is still trying to push an abortion ban and curb women’s rights. https://t.co/zTmJo9P2gy
    Tweet 317: A casual reminder that you don’t need to be an entrepreneur, founder, C-level exec or a manager for your work to ma… https://t.co/jcxAagr1cL
    Tweet 318: @kouky @benschwarz It’s for me, obviously
    Tweet 319: Seriously, @wnotw? Alcohol is the top perk you can think of? https://t.co/itaV4XwmOZ
    Tweet 320: RT @sarahcuda: this is well put. https://t.co/5sDOcqX29b
    Tweet 321: I need this pin immediately. https://t.co/EHTEFefzop
    Tweet 322: @dgoodlad might be of your interest https://t.co/pS2X5ExrLs
    Tweet 323: Less mentoring, more empowerment.
    Tweet 324: My life in a nutshell (plus a few great strategies on not talking about “being a woman in tech”).
    
    👏🏻 @vaurorapub!
    
    https://t.co/aTlTP0F8jI
    Tweet 325: Do yourself a favour and read this fantastic piece by @sw and @vaurorapub on getting free from toxic tech culture.
    
    https://t.co/CzPs8FaBcL
    Tweet 326: After running a 4-day conference and I’m now back on calling out exclusion and toxic tech culture.
    
    💁🏻‍♀️ 👋🏻
    Tweet 327: @benschwarz @adactio Also coffee at Karma, sweets at Massolit Bakery, wine at Charlotte, beer at Multi Qlti :)
    Tweet 328: RT @bubsolow: starting the job hunt ☞ would appreciate any junior developer roles / feedback on my resume https://t.co/PiSPoASsEf
    Tweet 329: @jennschiffer @laras126 @gr2m @jimthoburn @rouzbeh84 Yep, I just wanted to clarify why I was asking :)
    Tweet 330: @jennschiffer @laras126 @gr2m @jimthoburn @rouzbeh84 I understand that, however I’d like to point out that there ar… https://t.co/Bn3xxN1M2S
    Tweet 331: My event organizing performance review lies in community impressions. 🙏🏻 https://t.co/4mzFEwwzKR
    Tweet 332: @laras126 @gr2m @jimthoburn @rouzbeh84 @jennschiffer Hmmmm... did that go through the rules for running a CSSConf? https://t.co/KaFgIwo4Rd
    Tweet 333: I wish all of the brands capitalizing on the feminist movement at least had some baseline of inclusion.
    
    Make it in… https://t.co/yK3F8MMBP0
    Tweet 334: Looking to read some new, good product and design books. Any recommendations for publications within the last 3 years? 🙏
    Tweet 335: @tsunamino omg, so excellent!
    Tweet 336: @sch Sure! Thank you for being open to chatting. Emailed. :)
    Tweet 337: @sch I’m in Melbourne, which has 5 hours overlap with SF :)
    Tweet 338: @sch hey! I’m assuming this isn’t a remote job?
    Tweet 339: @jasonfried I always really enjoyed 37signals’ books, but I can’t unsee the ableism in the title...
    Tweet 340: I’d love to have more time to build up my professional portfolio but it’s hard to focus when there’s no job securit… https://t.co/QOnLi6v91M
    Tweet 341: As much as I love connecting with fellow women at tech events, it always breaks my heart so much that half of the c… https://t.co/DlZtj3F2AQ
    Tweet 342: @mapotato I’m so sorry about this, Theresa. We will try to find out the name of the offender. There will be consequences.
    Tweet 343: Inclusion sections on careers pages mean nothing if you can’t reply to all candidates no matter the outcome and giv… https://t.co/JrBPxHYF7m
    Tweet 344: @almonk @jgwhite @itchymutt Interesting. I’d be keen to chat in that case :)
    Tweet 345: RT @duyenho: Hats off to @Fox for being the real deal 💪 Thank you Karolina and @benschwarz for the last ever #jsconfau18 https://t.co/8oRoy…
    Tweet 346: RT @not_sherry_wine: Shoutout to @fox and @benschwarz for the fantastic #JSConfAU18. The opportunity program is one of the best gift I’ve b…
    Tweet 347: RT @the_patima: 📢✨ I have something to tell you all...
    
    I'm going to Berlin in June to speak at @jsconfeu!!!! 💖
    
    🌟 It's my 1st ever trip to…
    Tweet 348: @itchymutt @heroku SF-based?
    Tweet 349: RT @susanthesquark: I've said it a million times and I'll keep shouting it from the rooftops: a company's diversity and inclusion efforts m…
    Tweet 350: Apparently someone said that they’re only afraid of meeting two people; @dhh and myself.
    
    I will take this as a compliment.
    Tweet 351: That’s a wrap, everyone. CSSConf and JSConf Australia are now a thing of a past.
    
    Hope we left you inspired and with new friends.
    
    🙌
    Tweet 352: @robodana Thank you for your kind words, Dana! I’m glad you’ve enjoyed it
    Tweet 353: @kriesse @benschwarz Texting from bed, yes 🌸
    Tweet 354: @sentience Ah, interesting! Great photos 👏🏻
    Tweet 355: @sentience what camera and lenses are you using? Fuji or Leica? The quality is quite great.
    Tweet 356: RT @JSConfAU: Did you enjoy the live stream today? Great, because we’re doing it again today.
    
    Tune in for awesome tech talks. 🙏❤️ #jsconfa…
    Tweet 357: I have so many subtweets about people misbehaving in subtle ways at conferences I could probably write a book by now. 
    
    Last day to go!
    Tweet 358: RT @fox: 🚨👋🏻 Hi everyone! I’m searching for new opportunities. 
    
    I’m looking for multidisciplinary product, front-end and leadership roles.…
    Tweet 359: @RedRoxProjects Thank you! 🌺
    Tweet 360: @agisilaosts thank you! 🌺
    Tweet 361: We’re live streaming @jsconfau talks all day today and tomorrow. Starting in 30 minutes!
    
    https://t.co/ttIzoV4Cjz
    Tweet 362: RT @frameshiftllc: New angle for change in VC: founders refusing to take funding from all-male all-white firms https://t.co/AMi8UJbTGO
    Tweet 363: @sentience @benschwarz @glenmaddern @xzyfer @jordanlewiz I like how everyone is wearing black or tan pants.
    Tweet 364: @kriesse @benschwarz ❤️❤️❤️
    
    I’ll have a celebration of lying face down in a few days.
    Tweet 365: @butwhoiskat thank you!
    Tweet 366: Here it goes. Last 8 months of constant work will happen within the next 3 days.
    
    https://t.co/c48sQqdnXv
    Tweet 367: RT @cssconfau: Guess what?
    
    Even if you aren’t here, you can watch all the talks LIVE! 😱
    
    Tune into the live stream here. #cssconfau18 
    
    ht…
    Tweet 368: @amyngyn hahaha. I actually never been, but I’ve heard the bar is great. :)
    Tweet 369: @amyngyn Loui Bar?
    Tweet 370: RT @LJKenward: Hey friends! 👋 Who's hiring Junior Devs at the moment? I have some awesome people from the @juniordev_io Community currently…
    Tweet 371: Don’t forget about the Community Social today! EVERYONE IS WELCOME (even if you don’t hold a CSSConf or JSConf tick… https://t.co/6c79TcKFCi
    Tweet 372: Toxic tech industry creates a fake vision of what each of us (especially minorities) have to be and achieve to be ”… https://t.co/yfXZ5nxCJ7
    Tweet 373: @amyngyn I never introduce myself. The focus is on content, not myself. Also I don’t feel like I need to justify my cred. :)
    Tweet 374: Today I got kissed by a dingo. 💁🏻‍♀️ https://t.co/FDQsVw2anl
    Tweet 375: @Sareh88 Thank you, Sareh! That’s very kind. ❤️
    Tweet 376: @meelijane https://t.co/Y5wM3nCdsH in Northcote. I’ve tested many and this one is orders of magnitude better than everywhere else. :)
    Tweet 377: One of many reasons why I love my yoga studio so much is how meditative the practice is and how all the instructors… https://t.co/VXfSqdo6bk
    Tweet 378: @IvanaMcConnell I can only help ruin your bank account further, sorry. 😂
    Tweet 379: RT @slamup: people love to say
    
         no child is born 
         with hate in their heart
    
    which is all very
    romantic
    
    but from the moment
    a bla…
    Tweet 380: @evanderkoogh Nope, we are at full capacity of the venue. :)
    Tweet 381: RT @cssconfau: Come and celebrate with us at pre CSSConf and JSConf AU community social!
    
    📅 Monday, March 19, 6pm onwards
    📍Stomping Ground…
    Tweet 382: @noahmp Heh, dang :) worth asking nonetheless.
    Tweet 383: @noahmp 👋🏻 is this a SF-based role?
    Tweet 384: RT @mbrockenbrough: Here's a point worth making every so often. The patriarchy isn't men. It's a system that prefers them. Wanting to disma…
    Tweet 385: @madalynrose Thank you so much ☺️ looking forward to meeting you! 🌺
    Tweet 386: @andymcmillan Thanks, Andy! You are an inspiration for me too! 💙
    Tweet 387: @evanderkoogh Hey Erwin! Thanks so much. We can chat during the events. :)
    Tweet 388: I don’t know what or who I’m most disappointed with to allow community work put my career in the background (again)… https://t.co/khOCa8rLHV
    Tweet 389: This time was supposed to be split between the conference and product work that would set me up for looking for a j… https://t.co/mOoFPPnV5w
    Tweet 390: Over the last 6+ months, I’ve sacrificed all the time I had to run CSSConf and JSConf AU. I’ve set the highest stan… https://t.co/QRvns2lJs8
    Tweet 391: RT @katebevan: LAYDEEZ!!!! Worried that VPNs are too hard for your fluffy ladybrain??? Never fear, a fuckwitted BroCo called @keepsafe is h…
    Tweet 392: @sarah_edo thank you! 😳
    Tweet 393: I can’t wait to come back to lovey Portland and see what wonderful thing @andymcmillan and @waxpancake are preparin… https://t.co/3rfz3ShOEU
    Tweet 394: @jennwrites thank you! I miss you too 😭❤️
    Tweet 395: To the young woman wearing a “the future is female” tee:
    
    The
    Future
    Is
    Intersectional
    Tweet 396: Cannot agree with this more. I constantly get asked for free diversity, inclusion, community or general workplace a… https://t.co/M9hOz6A980
    Tweet 397: @pat @coryannj @kckal Oh, I have not seen it. Will register. 👍🏻
    Tweet 398: Four days to go. https://t.co/RYxiCmMEFp
    Tweet 399: Ellen already had a lasting impact on diversity and inclusion spanning beyond the tech industry. 
    
    I can’t wait to… https://t.co/1QFSP6DHyQ
    Tweet 400: @jordwalsh 👋🏻 interesting! Would you be able to email me more details? hi at https://t.co/vah0lKcYeo. 📬
    Tweet 401: RT @NYTStyles: "How can I get over my sense of betrayal, my rage and my desire to punish this man for the disrespectful way he treated me?"…
    Tweet 402: Detectives in Los Angeles searching for an actress who disappeared last month found a body they believe is hers, th… https://t.co/UctTxeFEo6
    Tweet 403: "We have created a double tragedy for these people." Chronic pain patients treated with high doses of opioids are c… https://t.co/O2ZPgi7gM5
    Tweet 404: RT @nytopinion: John Paul Stevens: Rarely in my lifetime have I seen the type of civic engagement school children demonstrated throughout t…
    Tweet 405: Before harvesting millions of users' Facebook data, Cambridge Analytica had help from an employee at Peter Thiel's… https://t.co/yq5T0MfyyH
    Tweet 406: Investors pummeled tech stocks again on Tuesday as they hurried to drop shares in the very sector that once drove a… https://t.co/55Ge3ZrTLI
    Tweet 407: RT @PamelaPaulNYT: Everyone except Sean Penn will enjoy reading this book review. https://t.co/RliFDhdK2o
    Tweet 408: In Opinion,
    Isabelle Robinson, a senior at Marjory Stoneman Douglas High School, writes, "The idea that we are to b… https://t.co/cVB0DGRzd0
    Tweet 409: Bill Cunningham left us one final gift: a memoir https://t.co/jNLxToxSrK
    Tweet 410: RT @nickconfessore: NEW w/@AllMattNYT &amp; @carolecadwalla: interviews &amp; documents reveal how an employee at Peter Thiel's Silicon Valley inte…
    Tweet 411: He was wrongfully convicted of rape and murder and spent 23 years behind bars. This week, he returned to work as a… https://t.co/I6ujPJ625a
    Tweet 412: RT @jdelreal: Hundreds of people who were waiting outside have made their way into the city hall foyer, where they are chanting Stephon Cla…
    Tweet 413: Breaking News: President Trump secured his first major trade deal: a pact with South Korea. It may have been driven… https://t.co/ie34rULVmY
    Tweet 414: Evening Briefing: Here's what you need to know at the end of the day
    https://t.co/viDImohrSe
    Tweet 415: Dr. John Cacioppo, who bridged biology and psychology in exploring the health effects of loneliness, dies at 66 https://t.co/Czq6evXOGX
    Tweet 416: RT @jdelreal: Stephon Clark’s brother is interrupting the city council meeting. “They don’t care about you!” he screamed. https://t.co/GrXi…
    Tweet 417: Some might find the idea of speed-solving @nytimes crossword puzzles intimidating. Not this group.… https://t.co/bfSPo3yCxa
    Tweet 418: At least 12 states signaled that they would sue to block the Trump administration from adding a question about citi… https://t.co/C4Pn782AFc
    Tweet 419: RT @jdelreal: Packed house today at the Sacramento City Council meeting to discuss the Stephon Clark shooting. The room is at capacity and…
    Tweet 420: The surprise discussions added another layer of complexity to the rush of global diplomacy around North Korea’s nuc… https://t.co/fkwmip5h3Z
    Tweet 421: Breaking News: North Korea's leader is said to have met secretly with China's president. It was Kim Jong-un’s first… https://t.co/Mw0YpWvfbC
    Tweet 422: As you walk into a room at University of California, Irvine, the first thing you notice are the fruit and vegetable… https://t.co/c7KTBOf2dZ
    Tweet 423: H&amp;M is a "fast fashion" giant. But it has a big problem: A $4.3 billion pile of unsold clothes. https://t.co/xkjuVRmCJT
    Tweet 424: It's pretty clear how birds, even dinosaurs, got their wings. But how insects got theirs has been a mystery -- unti… https://t.co/izgS59NtxG
    Tweet 425: Evening Briefing: Here's what you need to know at the end of the day
    https://t.co/v54i6wg4pj
    Tweet 426: RT @jdelreal: About two dozen people are gathered here at the Sacramento County District Attorney’s Office to protest the death of Stephon…
    Tweet 427: Apple is introducing a new iPad, trying to become a force again in classrooms, a battle fought against Google and M… https://t.co/AAPH4kWu1N
    Tweet 428: The 2020 census is a snapshot of America that will affect the number of congressional seats and how state and feder… https://t.co/HcGCEs7ftP
    Tweet 429: RT @cliffordlevy: Some Republicans fear the midterms may come down to one thing: Trump’s conduct. (Including Stormy.)
    "He blocks everything…
    Tweet 430: A fatal helicopter crash in New York’s East River may have been caused by a passenger’s harness accidentally trippi… https://t.co/ilkS6nx9rz
    Tweet 431: He was wrongfully convicted of rape and murder and spent 23 years behind bars. This week, he returned to work as a… https://t.co/zIGxzZlm4E
    Tweet 432: Here are 11 movies you won’t want to miss https://t.co/DeWdoxFniP
    Tweet 433: If President Trump actually meets Kim Jong-un, his challenge will be much larger than merely persuading North Korea… https://t.co/7JveTkU9oD
    Tweet 434: “The Americans” has always been as much about marriage and partnership as about geopolitics, our critic writes https://t.co/UXPe1hO5R1
    Tweet 435: Roseanne believes President Trump doesn't oppose same-sex marriage. "He has said it several times, you know, that h… https://t.co/E7ThXtFRAn
    Tweet 436: If it is successful, the Trump administration could come closer than any Republican White House has to achieving a… https://t.co/edKynbq7K7
    Tweet 437: These almond cookies need just 4 ingredients https://t.co/gY9TpfJdjd
    Tweet 438: In Opinion,
    Isabelle Robinson, a senior at Marjory Stoneman Douglas High School, writes, "The idea that we are to b… https://t.co/uxs0El114r
    Tweet 439: "A strong case could be made for [Gerald] Murnane, who recently turned 79, as the greatest living English-language… https://t.co/7ESwrnkrfv
    Tweet 440: Two graphic designers founded Turbo without fully knowing what it would be. Now, the studio is part of a growing de… https://t.co/xmZNfQqdZX
    Tweet 441: The attorney general is responding to public outcry over a police shooting in which an unarmed black man was killed… https://t.co/64nSNabytF
    Tweet 442: The photos are surreal, like a Martian ski slope or a toasted marshmallow sky https://t.co/VFHIgBVETa
    Tweet 443: By branching into public forums, Breitbart is taking a cue from one of its perennial foes: the mainstream media https://t.co/8wS7IUFeQU
    Tweet 444: Check out our new animations about race and mobility, or make one of your own https://t.co/FNt0PtuI8l
    Tweet 445: Horrific accounts of children struggling to escape the blazing shopping mall distracted public attention from a dip… https://t.co/0ngky3p6Km
    Tweet 446: Fair housing groups filed a lawsuit saying that Facebook continues to discriminate against certain groups, includin… https://t.co/cBbsa2X9BK
    Tweet 447: RT @nytimesworld: “Spain is creating a situation where Europe’s judges rather than its politicians are being asked to solve Catalonia,” one…
    Tweet 448: Under pressure, Mark Zuckerberg agreed to testify before Congress over Facebook’s handling of user data, people fam… https://t.co/MvIygDFq9u
    Tweet 449: Critics say the rule could throw patients who lost access to the drugs into withdrawal or even provoke them to buy… https://t.co/06RkHiijim
    Tweet 450: Tempest in an egg spoon: How Alice Waters's fancy utensil set off a culinary culture war that touches on class and… https://t.co/g0XvIB6ABy
    Tweet 451: Lieutenant Davidson left behind a wife and four young children, and came from a family of firefighters, with his fa… https://t.co/qoREudBKjp
    Tweet 452: “It’s like a beauty pageant. The fish cannot be fat. It must look strong and have personality." https://t.co/nSKiuU0x9i
    Tweet 453: RT @amyfiscus: "Bulletproof, Slow and Full of Wine” is also the title of my autobiography https://t.co/JthIHU4sY5
    Tweet 454: Roseanne has become a Trump supporter https://t.co/bs6NXUo6mr
    Tweet 455: Tanzina Vega is the new host of “The Takeaway,” succeeding John Hockenberry, who was accused of sexual harassment a… https://t.co/zrO7bNE7I5
    Tweet 456: Why are the Parkland students being attacked? “Together we kind of form an unstoppable force that terrifies them." https://t.co/wTnnc2IOLq
    Tweet 457: RT @KevinQ: New today in Upshot-land, income mobility ladders for girls, Asian-Americans and other groups. Or make your own – there are tho…
    Tweet 458: "I got back to something I’d been missing over a month of solo travel: the joy of getting lost with someone whose c… https://t.co/uTaaAFEJSa
    Tweet 459: "Being crazy isn't enough." A Manhattan nanny charged with killing two children in her care is pursuing an insanity… https://t.co/ro78pJw85B
    Tweet 460: Two white police officers in Baton Rouge, La., will not be prosecuted in the fatal 2016 shooting of Alton Sterling,… https://t.co/kRzukrAMDy
    Tweet 461: The former dean, William D. Strampel, was accused not only of failing to protect women and girls from Larry Nassar,… https://t.co/saN2IbTMhc
    Tweet 462: RT @nytimesbusiness: “Trump has been a godsend for China.” The potential cost of moving away from institutions and alliances in favor of a…
    Tweet 463: The Rev. Samuel Rodriguez thinks of himself as a modern-day Joseph in Pharaoh’s court, placed there to save his peo… https://t.co/ajpHlmGwE4
    Tweet 464: RT @patrickhealynyt: Roseanne Barr &amp; I got into it when she claimed Trump favors same-sex marriage: "Yes, he does. He has said it several t…
    Tweet 465: Federal authorities charged Keith Raniere, the head of the Albany-area group Nxivm, with forcing women to have sex… https://t.co/L5xGmbNUfG
    Tweet 466: In Opinion,
    Op-Ed columnist @PaulKrugman writes, "The simple truth is that ever since Reagan, Republicans have basi… https://t.co/2xROYOSesk
    Tweet 467: RT @nytopinion: Stormy Daniels is not the first person who claims to have been threatened after crossing Donald Trump https://t.co/jq4XA1hs…
    Tweet 468: RT @nytimesarts: Chris Evans makes a terrific Broadway debut in "Lobby Hero" https://t.co/RSHFjyp02o
    Tweet 469: On U.S. involvement in the Syrian war, one operative said: “Russia is more honorable and trustworthy than the Unite… https://t.co/j4nPJbWm4Q
    Tweet 470: RT @nytimesbusiness: Waymo says it will buy up to 20,000 electric cars from Jaguar Land Rover as it strives to put a ride service into oper…
    Tweet 471: Stephen Colbert and other late-night hosts were not surprised by Stephanie Clifford's "60 Minutes" interview https://t.co/i7tEE6rroE
    Tweet 472: "I'm sure there are other Cambridge Analyticas out there," Senator John Kennedy said. "Facebook isn't just a compan… https://t.co/q02W3HcwDe
    Tweet 473: RT @nytopinion: John Paul Stevens: Repealing the Second Amendment would move Saturday’s marchers closer to their objective than any other p…
    Tweet 474: Polls and recent elections show that Trump has galvanized liberal and moderate voters to oppose his party. Yet at t… https://t.co/hj0feZkpk5
    Tweet 475: RT @nytgraphics: All 435 House seats are up for election this November, but just 48 are considered competitive: 41 held by Republicans, 7 b…
    Tweet 476: Morning Briefing: Here's what you need to know to start your day https://t.co/1cL3jRvE6f https://t.co/DZGs6ucJlW
    Tweet 477: The Maryland school shooter who fatally shot his ex-girlfriend and injured another student last week killed himself… https://t.co/n7N1XpSNsB
    Tweet 478: Linda Brown came to symbolize one of the most transformative court proceedings in American history https://t.co/vUcEP99A3g
    Tweet 479: A housemate of the Austin bomber has become a "person of interest" in the investigation https://t.co/thNrBdpcJM
    Tweet 480: Your daily @DealBook Briefing:
    
    • Citigroup became the biggest Wall Street firm thus far to take actions to limit g… https://t.co/trOfabkKIY
    Tweet 481: Lobster, wine and "lady conductors": What we know about North Korea's mystery train https://t.co/PCuXFczwwb
    Tweet 482: "From a European perspective, the shock comes from the fact that the U.S. is now seen as a destabilizing force, lik… https://t.co/IMDcz7Nj2v
    Tweet 483: Eight years ago, the United States and Russia agreed to a spy swap that sent a Russian double agent to safety in Br… https://t.co/5Dv7DOef5y
    Tweet 484: The former dean of Michigan State University’s medical school, who supervised the disgraced physician Lawrence Nass… https://t.co/zvJMTgrnB7
    Tweet 485: Within 30 seconds of arriving, Deputy Brewer had exited his car, confronted a man in the street whose pants were ar… https://t.co/LzNamOCMQf
    Tweet 486: The 2020 census will ask respondents whether they are U.S. citizens, the Commerce Department announced, agreeing to… https://t.co/zz1aqLKqAT
    Tweet 487: A video that appeared to show the arrival in Beijing of an old-style green train fueled speculation that a high-lev… https://t.co/nB2RJVVDYe
    Tweet 488: Morning Briefing: Here's what you need to know to start your day https://t.co/C4v9hpi3Ye https://t.co/QLK2dlg5qA
    Tweet 489: Others have come at President Trump with indignation, righteousness and appeals to decency. Stormy Daniels swatted… https://t.co/iq3gZcZzNd
    Tweet 490: "She called me at 4:11 p.m. the last time and told me that everything was in flames, and that the doors were blocke… https://t.co/T2XmtaoKxp
    Tweet 491: A woman who survived the Holocaust was murdered in Paris, in what the authorities are calling a hate crime https://t.co/PCfeDQFPrz
    Tweet 492: After 61 weeks in the White House, President Trump has found 2 people he won’t attack on Twitter: Stormy Daniels an… https://t.co/5VpaR8vviN
    Tweet 493: China's first space station, Tiangong-1, abandoned and out of control, is expected to drop out of orbit around this… https://t.co/IURCPOQDHv
    Tweet 494: RT @nytimesworld: To Europeans, the brazen poisoning of a former Russian spy and his daughter in Salisbury, England, crossed a line. That h…
    Tweet 495: The main Brexit campaign in the referendum on Britain’s EU membership funneled more than $900,000 to a puppet organ… https://t.co/9Vzlfx7fHr
    Tweet 496: President Trump has stayed in touch with Rob Porter and has told some advisers he hopes Porter returns to work in t… https://t.co/AM9RjUNwtB
    Tweet 497: A sprawling exhibition in Amsterdam looks at how a fascination with Japan shaped van Gogh's work https://t.co/1kMpdRcvCs
    Tweet 498: “The world’s patience is rather wearing thin with President Putin and his actions,” said the British defense secret… https://t.co/mfKxquWuau
    Tweet 499: RT @meslackman: All roads may lead to Rome, but when you get here the mean streets and wrecked pavements will puncture your tires, break yo…
    Tweet 500: The Ethicist: Must I Tell My Boss I’m Absent Because of Mental Illness? https://t.co/OmG8vR4vhG



```python
sentiments_df = pd.DataFrame.from_dict(sentiment_array)
sentiments_df['Media Source'] = sentiments_df['Media Source'].map(lambda x: x.lstrip('@'))
```


```python
# Data Frame for holding sentiments
sentiments_csv = sentiments_df[['Media Source','Date','Text','Compound','Positive','Neutral','Negative','Tweet Count']]
sentiments_csv.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media Source</th>
      <th>Date</th>
      <th>Text</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Neutral</th>
      <th>Negative</th>
      <th>Tweet Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>Tue Mar 27 18:30:07 +0000 2018</td>
      <td>When mother Marie mysteriously leaves the fami...</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC</td>
      <td>Tue Mar 27 18:00:08 +0000 2018</td>
      <td>🇩🇪😂 Even if you don't speak German, this is wo...</td>
      <td>0.2942</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC</td>
      <td>Tue Mar 27 17:00:07 +0000 2018</td>
      <td>🍜 We've got oodles of noodles with recipes for...</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC</td>
      <td>Tue Mar 27 16:00:15 +0000 2018</td>
      <td>😬 What does Facebook know about you? https://t...</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC</td>
      <td>Tue Mar 27 15:40:40 +0000 2018</td>
      <td>RT @BBCTwo: Happy #WorldTheatreDay! *leaves th...</td>
      <td>0.6114</td>
      <td>0.250</td>
      <td>0.750</td>
      <td>0.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiments_csv.to_csv("500 News Mood Tweets.csv")
```


```python
# Setting colors for each organization 
colors = {"BBC": "lightblue", "CBS":"green", "CNN":"red", "Fox":"blue", "nytimes": "yellow"}
```


```python
# Build the scatter plots for each media source 
plt.figure(figsize = (15,10))

for targets in colors.keys():
    Plot_DF = sentiments_df[sentiments_df['Media Source'] == targets] 
    plt.scatter(Plot_DF["Tweet Count"],
                Plot_DF["Compound"], 
                label = targets, color = colors[targets],
                edgecolor = "black", s=125)
    
plt.legend(bbox_to_anchor = (1,1), title = 'Media Sources')    

# Incorporate the other graph properties
plt.xlabel("Tweets Ago",fontweight='bold')
plt.ylabel("Tweet Polarity",fontweight='bold')
plt.title("Sentiment Analysis of Media Tweets (%s)" % (time.strftime("%x")),fontweight='bold')
plt.xlim(102,-2, -1)
plt.ylim(-1,1)
plt.grid(True)
sns.set()

# Save the figure
plt.savefig("SentimentAnalysis.png")

# Show plot
plt.show()
```


![png](output_8_0.png)



```python
#Mean scores by organization
scoresbyorganization = sentiments_csv.groupby("Media Source")["Compound"].mean()
scoresbyorganization 

```




    Media Source
    BBC        0.096663
    CBS        0.353884
    CNN       -0.010772
    Fox        0.273663
    nytimes   -0.061596
    Name: Compound, dtype: float64




```python
x_axis = np.arange(len(scoresbyorganization))
```


```python
# Build the bar chart for each media source 
plt.figure(figsize = (10,8))

for targets in colors.keys():
    Plot_DF = sentiments_df[sentiments_df['Media Source'] == targets] 
    plt.bar(x_axis, scoresbyorganization, color = {"lightblue","green", "yellow", "red", "blue"}, label = targets, edgecolor = "black")
    
plt.ylim(-.1, .45)
plt.ylabel("Tweet Polarity",fontweight='bold')
plt.axhline(y=0, color = 'black')
plt.title("Overall Media Sentiment Based on Twitter (%s)" % (time.strftime("%m/%d/%Y")),fontweight='bold')
x_labels = ["BBC", "CBS", "CNN", "Fox", "nytimes"]
x_locations = [value for value in np.arange(6)] 
plt.xticks(x_locations, x_labels)
sns.set()

# Save the figure
plt.savefig('Overall Media Sentiment Based on Twitter.png')

# Show plot
plt.show()
```


![png](output_11_0.png)

