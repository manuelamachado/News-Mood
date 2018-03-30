
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

From the sentiment analysis results with the five media tweets on March 30, 2018, we can note that:
 
Overall sentiment polarity is positive for BBC, CBS and Fox tweets and negative for CNN and NY Times.

From the overall media sentiment based on tweets, CBS is most positive at 35% and Fox News is the second most positive with 29%.

The results also show that CNN is most negative with 12% negative polarity and NY Times is second with 6% negative polarity, which the closest to neutral.

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

    Tweet 1: 'Change doesn't come from outside.'
    
    #Pilgrimage: The Road to Santiago | 9pm | @BBCTwo https://t.co/N4Fxc0EU77
    Tweet 2: ❤️ A dying man was granted his final wish in hospital - a visit from his dog. https://t.co/xE4rkcBJNv
    Tweet 3: The illegal wildlife trade is worth £18bn per year, but @wildliferescue are working to turn it around.… https://t.co/fe3MDr2TN9
    Tweet 4: RT @bbceurovision: 🇬🇧 Presenting our 2018 dream team! @Grahnort hosts the Grand Final on @BBCOne. Fresh from You Decide, we welcome @Rylan!…
    Tweet 5: RT @BBCNews: Heartbeat actor Bill Maynard dies at 89 https://t.co/BMeMLwjPpV
    Tweet 6: We use an estimated 8.5b plastic straws every year in the UK and they're among the top 10 items found in beach clea… https://t.co/K9r6HRVu3V
    Tweet 7: Graphic artist Gabriella Marcella uses a vintage #Risograph printer to produce amazing designs. https://t.co/W9JK5vLbSf
    Tweet 8: Meena was raised to be a ‘good Indian girl' - to cook, clean and find a husband.
    
    But as an environmental activist… https://t.co/HWMRk97Wdm
    Tweet 9: RT @BBC6Music: Elementary! Martin Freeman and @ACIDJAZZRECS' @eddiepiller delve into their record bag(gins) to share some choice cuts as th…
    Tweet 10: 🐘🤷‍♀️ A video of a 'smoking' wild elephant in India is baffling scientists and wildlife experts around the world. https://t.co/CENdAQYy5v
    Tweet 11: Advertisers are the experts at persuading us to eat burgers, crisps and fizzy drinks. But could they convince us to… https://t.co/JpuBKADOMB
    Tweet 12: It's been described as a 'game changer' for Multiple Sclerosis patients - scientists say a stem cell transplant can… https://t.co/Ru1R67FHig
    Tweet 13: 🍍 A squeeze of pineapple juice works wonders for dough...
    
    🍞 Here are 10 things you knead to know about bread! 
    
    👉… https://t.co/ajVBvN31mi
    Tweet 14: 😢🐘 This poor elephant has his foot caught on a deadly snare. 
    Can the team save him before its too late?… https://t.co/7pbk2WA7hW
    Tweet 15: At a time when men thought women shouldn't speak until they were spoken to, this female artist refused to be silent… https://t.co/mbaGBW4xMj
    Tweet 16: RT @BBCBreakfast: Counting down the hours until you go home for Easter?
    🌧🌤🌨 Here's your Easter weekend forecast: https://t.co/EXeBCHutS4
    Tweet 17: RT @bbcrb: Retirement's looking rosy for Hank the police horse! 🐎😆 https://t.co/xB9kXZeWNE
    Tweet 18: RT @BBCNewsbeat: Anthony Joshua faces Joseph Parker in Cardiff on Saturday.
    And he's eating a LOT to prepare... 😋 💪 https://t.co/qp6xw04IiC
    Tweet 19: RT @BBCOne: Anyone else having a 'chilled one'? 🐼 https://t.co/WRBeDdaK18
    Tweet 20: ❤️ 'That was it I just totally fell in love with art.'
    
    🎶 How grime changed Ade Adesina's life. https://t.co/eDyh75YBvl
    Tweet 21: 🍳 Wanna get the kids cooking this #EasterWeekend? 
    Here are 10 recipes to keep them busy in the kitchen.
    👉… https://t.co/5xxQNyyRMg
    Tweet 22: RT @BBCTaster: We've just released a host of amazing new artefacts for the #CivilisationsAR app! Discover history's treasures wherever you…
    Tweet 23: ✊ Plastic protest! https://t.co/9m59wayYH8
    Tweet 24: RT @bbcarts: This Picture Wall in Lahore, Pakistan stretches for over 450 metres and is full of mesmerising mosaics https://t.co/ZFhuF7PiQP
    Tweet 25: RT @bbcwritersroom: #OrdealByInnocence begins on @BBCOne on Easter Sunday 1st April at 9pm and @BBCiPlayer. #Screenwriter @PhelpsieSarah ex…
    Tweet 26: RT @bbcideas: Author Lionel Shriver says we project ourselves on to our possessions – and that she is her toy donkey, Clippety 😊.
    
    We'd lik…
    Tweet 27: RT @BBCTwo: Imagine living in this incredible Swiss chalet! 😍🇨🇭 #TheWorldsMostExtraordinaryHomes https://t.co/NofijuERJg
    Tweet 28: 😂 So, who's going out tonight? https://t.co/u2L4sEKjE2
    Tweet 29: 🎹🎶 Why it's never too late to learn an instrument. 
    👉 https://t.co/lbBppX8XlO
    
    #PianoDay https://t.co/x1CegC7iho
    Tweet 30: RT @bbcpress: Everyone's favourite fox reads out #MeanTweets (if he can get the right app open). For more @RealBasilBrush watch #Generation…
    Tweet 31: Happy birthday to Alfred and Robert! 🎈 🎉 Britain's two oldest men, born on the same day, have been celebrating turn… https://t.co/R7GXbEXl3v
    Tweet 32: This bizarrely cute creature is just the latest species to suffer a social media selfie ban. 
    👉… https://t.co/bG9qcXtouW
    Tweet 33: RT @BBCRadio2: Today we say farewell to the brilliant @RealLynnBowles. 😢
    Her sharp wit, warm tones, and encyclopaedic knowledge of the UK's…
    Tweet 34: 🐶❤️ Biscuit the robotic pup is helping the elderly cope with conditions like dementia. What a good boy! https://t.co/5vpg6KS031
    Tweet 35: 👀🌸 The world famous cherry blossoms of Japan bring hundreds of thousands of visitors to see them and this year they… https://t.co/kGMsyZnWzi
    Tweet 36: What do you get if you cross a polar bear and a grizzly? 
    
    🐻😍… A pizzly! 
    #DavidAttenboroughsNaturalCuriosities https://t.co/FwqiMiTd2m
    Tweet 37: 🇮🇪 This 800-year-old hotel in Ireland is simply incredible.  #AmazingHotels https://t.co/OAKKCsexPW
    Tweet 38: 📚 'Words do not live in dictionaries. They live in the mind.'
    
    Hear the only surviving recording of #VirginiaWoolf,… https://t.co/Puwg8Mve7N
    Tweet 39: The government rates the global outbreak of a deadly flu virus as a major threat to the UK. But how are flu vaccine… https://t.co/wzMylfDBrP
    Tweet 40: Slacking off in Dubai. https://t.co/a3oSKIxfSK
    Tweet 41: RT @BBCTheOneShow: The fantastic @george_ezra warming up with a quick cuppa after sound check. Join us at 7pm where he’ll be performing liv…
    Tweet 42: Two-thirds of drivers say they are "regularly dazzled" by oncoming headlights. 🚘 https://t.co/gwSgpsP6xx https://t.co/i9m9ph8QM1
    Tweet 43: The Masked Plumber is back. 
    And this time, he wants to show you how to unblock a sink trap. https://t.co/t1NIgRpZnO
    Tweet 44: RT @bbceurovision: We are starting a #Eurovision 2018 group chat. The tea from when @surieofficial met @saaraaalto 👉 https://t.co/c3PuiYT6S…
    Tweet 45: RT @BBCR1: "Last night I did have a very, very large bra, and it was very, very sweaty."
    
    @JaredLeto lets @Grimmers know about what gets th…
    Tweet 46: RT @BBCRadio4: Listen to the first three episodes of The Hitchhiker's Guide to the Galaxy: Hexagonal Phase now.
    
    https://t.co/GdvQZf3ijv
    
    #…
    Tweet 47: RT @bbcworldservice: Do you say fireman or firefighter? The woman in charge of the London Fire Brigade, Dany Cotton, says 'fireman' is sexi…
    Tweet 48: RT @BBCBreakfast: Take a look at this if you're wondering WHO PAYS for running the new plastic bottle and can deposit scheme: https://t.co/…
    Tweet 49: RT @BBCNewsround: Some advice from children on what NOT to say to people with an #invisiblecondition https://t.co/mGUj3TWB2J #Dynamo https:…
    Tweet 50: ♻️👛 People in England will soon have to pay a deposit when they buy drinks bottles and cans in a bid to boost recyc… https://t.co/qyKBqES5b3
    Tweet 51: 😲 Scientists in Australia have discovered the remains of a mummy in a 2,500-year-old coffin that was previously cla… https://t.co/2S4q2mDImu
    Tweet 52: RT @bbcthree: In our latest doc, we investigate allegations that R Kelly has been training a 14-year-old girl to be one his sex "pets" http…
    Tweet 53: Tickling sticks have been placed on statues in Liverpool ahead of Sir Ken Dodd's funeral. https://t.co/pPGGg07ZQi https://t.co/LSwzv5Qipw
    Tweet 54: 🛁🐤 Are your bathroom toys in-fact DEADLY? 😱 https://t.co/YrtMOzTvrb
    Tweet 55: RT @bbcarts: Andrew Scott, as Hamlet, performs the opening lines of #Shakespeare's most famous soliloquy. Watch the full production at 9pm,…
    Tweet 56: ❤️ After seeing her brother's struggles with depression, Luli's on a mission to break the stigma in the black commu… https://t.co/kl1ak4Gu6w
    Tweet 57: 😤 Feeling moody? 
    Your gut bacteria might be to blame...
    👉 https://t.co/2oPWKjf4D2 https://t.co/Xr5MwczfGR
    Tweet 58: 'I made a decision to not be unbelievably busy.'
    
    @Meshel_Laurie on beating busyness with Buddhism. https://t.co/7ynSgadRyq
    Tweet 59: When mother Marie mysteriously leaves the family home, the repercussions are enormous.
    
    #ComeHome | 9pm | @BBCOne |… https://t.co/S4JgnzQlAx
    Tweet 60: 🇩🇪😂 Even if you don't speak German, this is worth watching until the end! 
    #LiveAtTheApollo https://t.co/J3l8oToRbk
    Tweet 61: 🍜 We've got oodles of noodles with recipes for pad Thai, chow mein, ramen, pho and stir-fries.
    👉… https://t.co/JWfrcIhn2X
    Tweet 62: 😬 What does Facebook know about you? https://t.co/lG4ffeCG09
    Tweet 63: RT @BBCTwo: Happy #WorldTheatreDay! *leaves this here and runs away* @SaraPascoe 🎭😂 https://t.co/3XlEb5gr15
    Tweet 64: RT @BBCScotland: Meet the master of the radiator harp
    
    @DaftLimmy returns to BBC Scotland on Thursday 5 April. https://t.co/G7mjFEJx6k
    Tweet 65: RT @bbcthree: 12-year-old Keilan has very severe curvature of the spine. He needs surgery ASAP but the 5 hour operation could leave him par…
    Tweet 66: RT @bbccomedy: Cathy, master of shade. #Mum https://t.co/VjC2ttkaNF
    Tweet 67: RT @BBCR1: (•_•)
    &lt;)   )╯who
     /    \ 
    
      (•_•)
     \(   (&gt; bit
      /    \
    
     (•_•)
    &lt;)   )╯Beyoncé? 
     /    \ 
    
    https://t.co/7hQd9mcKUY
    Tweet 68: RT @BBCWorld: It's 50 years since the death of Yuri Gagarin, the Russian cosmonaut who was the first man to travel into space 👨‍🚀 https://t…
    Tweet 69: ❤️ The first polar bear cub to be born in the UK for 25 years has been filmed adapting to its surroundings. https://t.co/EsxbH18cN5
    Tweet 70: The ancient Greeks thought a life of ‘constant leisure’ was the highest life you could live. https://t.co/sascCq8iks
    Tweet 71: There's over 50 classic cookery shows being served up on @BBCiPlayer this #EasterWeekend. 
    👉 https://t.co/CKECZMGmx8 https://t.co/j3RfAYQvTI
    Tweet 72: 🎭 @MargotRobbie is planning a new TV series, which will give Shakespeare plays a 'female perspective'.
    👉… https://t.co/pTLXfve0ds
    Tweet 73: 🚀🌎 'Mad' Mike Hughes flew his home-made rocket to try and 'prove' the earth is flat. 
    
    🚨 SPOILER: it's not.  https://t.co/f5oNRItwyq
    Tweet 74: RT @BBCTwo: Now we've got our country back... what actually is it? 🤔
    
    #CunkOnBritain starts Tuesday 3 April, 10pm, @BBCTwo. @missdianemorga…
    Tweet 75: RT @BBCBreakfast: A team of abseilers have been roped in to give Cheddar Gorge in Somerset an extreme spring clean. https://t.co/iYmkiizeRi
    Tweet 76: RT @BBCR1: We can't get enough of @george_ezra's Live Lounge 😍
    
    Watch him perform 'Paradise' and cover 'These Days' right here 👉https://t.c…
    Tweet 77: RT @bbcgetinspired: Check out @JesseLingard and some of his @PremierLeague mates showing us their super moves. 🕺🏻 
    
    Show us yours with #sup…
    Tweet 78: When this woman visited an uninhabited Caribbean island, the last thing she expected was to find it covered in plas… https://t.co/cpKW9tl61v
    Tweet 79: About 700,000 people in the UK are on the autism spectrum, with five times as many males as females being diagnosed… https://t.co/PXJMuG76Tn
    Tweet 80: The number of children living in relative poverty in the UK has risen to 4.1m. 
    
    This is the story of Tyler, one of… https://t.co/xFtEAjBTXC
    Tweet 81: Filmed in Nottingham during the worst winter for the NHS on record, groundbreaking series #Hospital returns tonight… https://t.co/PVSIcvCGc9
    Tweet 82: Tonight, George Alagiah explores the fascinating history of Queen Elizabeth II and her beloved Commonwealth. 
    
    The… https://t.co/xXer4FRyTq
    Tweet 83: 🐶😂 It was a Dachsh(und) to the finish line. https://t.co/7VvkU5GfUg
    Tweet 84: ✈️ Joy and Mary flew WW2 planes before any navigation system was installed.
    #RAFat100 https://t.co/yxNMblfTEf
    Tweet 85: Tom Cruise stars in a remake of the 1953 adaptation of HG Wells’s classic novel. 🍿
    
    War of the Worlds | @BBCiPlayer… https://t.co/h19JvTlFSA
    Tweet 86: In her role as head of the Commonwealth, the Queen embarked on her first tour of the nations in 1953.
    
    But by 1970… https://t.co/Av61bahNZF
    Tweet 87: Never let a squirrel nibble your nobble... 🎈🐿😱
    https://t.co/cHm9m6At3m
    Tweet 88: Apple wants to introduce new emojis to better represent people with disabilities. ❤️️ https://t.co/CcJfojmtTa https://t.co/ZRdEhVPMEm
    Tweet 89: A crack that opened up in Kenya’s Rift Valley, damaging a section of the Narok-Nairobi highway, is still growing... https://t.co/T5YocDauYj
    Tweet 90: From hot cross bun gin to Creme Egg Yorkshire pudding, this year’s Easter treats are bigger, weirder and more decad… https://t.co/HvDJmgcB5q
    Tweet 91: RT @bbcpress: Sue Perkins will return to host the 2018 #BAFTA TV awards on Sunday 13 May on @BBCOne. https://t.co/pieSWNAGbH
    Tweet 92: 😂 @BillBailey is NOT a fan of taramasalata. #Room101 
    
    https://t.co/soAAz27c0C
    Tweet 93: The kiwi bird's egg is so large, it takes almost ten days to lay! 🐦🥚😳  #DavidAttenboroughsNaturalCuriosities
    https://t.co/AkZqK9vzZZ
    Tweet 94: 🤔Why do so many celebrities decide to enter politics?
    
    Some have done so more successfully than others...… https://t.co/MkEDpSeLMJ
    Tweet 95: Wishing you could get out of bed just that little bit earlier? 😴
    
    If you need some motivation to set that alarm, he… https://t.co/zsisa2oA4s
    Tweet 96: Could this be an answer to global water shortages? 🏜💧 This machine creates water out of thin air. 
    
    https://t.co/caz4nXMJg5
    Tweet 97: Tonight, @regyates meets people whose lives have been devastated by the Grenfell fire. 
    
    Reggie Yates: Searching fo… https://t.co/HPgtcZuHte
    Tweet 98: Tonight, @mcgregor_ewan and @McgColin celebrate the centenary of the Royal Air Force. 
    
    RAF at 100 with Ewan and Co… https://t.co/nF2iwBP51b
    Tweet 99: The first ever statue of David Bowie has been unveiled in the town where he debuted Ziggy Stardust. ⚡️… https://t.co/lFgROYVkv1
    Tweet 100: When you're enjoying being single and people just can't deal with it. 🙄😂 @kathbum #LiveAtTheApollo 
    
    https://t.co/byHMHWyhPq
    Tweet 101: Take and post a photo of the woman in your life who inspires you! The photo should only have one person against a n… https://t.co/BHdxXlWfKP
    Tweet 102: Expect epic performances by @kanebrown , @kelly_clarkson, @OfficialJackson, and more at this year's 53rd #ACMawards… https://t.co/3yso7Z0cjv
    Tweet 103: Will @HIGHVALLEY, @LancoMusic, @LOCASHmusic, @MidlandOfficial, or @runawayjune be named New Vocal Duo Or Group Of T… https://t.co/v0QIPLC47R
    Tweet 104: Count on Entertainer Of The Year nominee @LukeBryanOnline to crash the party with an epic performance at the 53rd… https://t.co/27ua60HTqu
    Tweet 105: Join @eltonofficial and some of today's hottest names in music when they take the stage to perform his most memorab… https://t.co/jz0jaShZIj
    Tweet 106: RT @ACMawards: The ACM for New Vocal Group of the Year goes to @MidlandOfficial! And yes, that really was @Reba on the phone! #ACMawards ht…
    Tweet 107: RT @ACMawards: In case you didn’t know, the ACM for New Male Vocalist of the Year goes to @BrettYoungMusic. Check out his reaction when @Re…
    Tweet 108: RT @ACMawards: Over the weekend @Reba called the ACM New Artist of the Year winners to let them know they had won! Let’s just say our New F…
    Tweet 109: Congratulations to the 53rd #ACMawards New Artist winners @Lauren_Alaina, @MidlandOfficial, and @BrettYoungMusic! W… https://t.co/Fr8H4arwGj
    Tweet 110: New start times in East/Central Time Zones #60Minutes 7:35ET/6:35CT #Instinct  8:35ET/7:35CT #NCISLA 9:35ET/8:35CT… https://t.co/8W5hAeLrvs
    Tweet 111: Don’t miss a minute of the action. Stream the Elite Eight® games LIVE today starting at 2PM ET with a FREE trial of… https://t.co/8NwU8HdiHR
    Tweet 112: RT @MomCBS: That's a wrap on the #Mom panel at #PaleyFest! Thanks for following along! https://t.co/we4JgqPt6P
    Tweet 113: RT @MomCBS: A fan just commented that #Mom helped bring him out of a deep depression. 💜💜💜 #PaleyFest
    Tweet 114: RT @MomCBS: "Go out for it anyway. If you're good for the role, you're good for the role." @theJaimePressly's advice for aspiring actors wi…
    Tweet 115: RT @MomCBS: Mom Co-Creator @GemmaRBaker just pointed out her own #Mom in the audience at #PaleyFest! 💜
    Tweet 116: RT @MomCBS: "I'm not someone in recovery who goes to AA, but I have taken so much away from it...to take one day at a time." - @theJaimePre…
    Tweet 117: RT @MomCBS: "You get to appreciate working with such talented people." - @AnnaKFaris #Mom #PaleyFest
    Tweet 118: RT @MomCBS: “I love this job. I love working with these women. I love working in front of the live audience… It’s alive and it’s fun.” - @A…
    Tweet 119: Get on your feet for @Jason_Aldean, @ThomasRhett, @ChrisStapleton, @KeithUrban, and @ChrisYoungMusic, the five nomi… https://t.co/oT5ogjdj4x
    Tweet 120: Get ready for some sweet games! Stream #5 Clemson vs #1 Kansas LIVE at 7PM ET and #11 Syracuse vs #2 Duke LIVE at 9… https://t.co/4WstgrKNnW
    Tweet 121: RT @SEALTeamCBS: In honor of #NationalPuppyDay... 😍 #SEALTeam https://t.co/4mIZPpiRlU
    Tweet 122: RT @HawaiiFive0CBS: Nothing like a man and his dog! 😍🐶 Happy #NationalPuppyDay to Eddie, the best pup on the Five-0 Task Force! #H50 https:…
    Tweet 123: Game on! 16 teams left and the race to the finish continues tonight. Stream #11 Loyola-Chicago vs #7 Nevada LIVE at… https://t.co/W374rmzzoC
    Tweet 124: Save the date! These are season finales you do NOT want to miss. RT if you're excited! https://t.co/UUQoWsPPSh https://t.co/cDl4WmxMtU
    Tweet 125: Congratulations to all of the @CBSDaytime nominees for the #DaytimeEmmys! See the full list of #DaytimeEmmy nominee… https://t.co/ivJVJWvfsf
    Tweet 126: Female Vocalist Of The Year nominee @MarenMorris will show her fans how it’s done when she takes the stage to showc… https://t.co/PITjmAoFT8
    Tweet 127: The legendary @Reba returns to host the 53rd #ACMawards and she’s proving just how comfortable she is behind the mi… https://t.co/XPXcSPRXqC
    Tweet 128: RT @nancyodell: Told my daughter I'd be presenting at @ACMawards again this year. (Woot woot!We both luv country music!)She took this pic o…
    Tweet 129: RT @ladyantebellum: Ecstatic to announce we'll be performing at the #ACMawards in Las Vegas again this year! https://t.co/Qfhs94j6FR
    Tweet 130: Country superstars @kennychesney, @ladyantebellum, @blakeshelton, and @KeithUrban have just been added to the stell… https://t.co/bJ4If7MacP
    Tweet 131: RT @YandR_CBS: Forever evolving, Forever inspiring, Forever Young and Restless. ❤️ Get ready to celebrate 45 years of #YR starting in just…
    Tweet 132: New start times in East/Central Time Zones: #60Minutes 7:37ET/6:37CT #Instinct series premiere 8:37ET/7:37CT… https://t.co/xT3YKqmu2M
    Tweet 133: Spend your Sunday streaming Second Round games LIVE with a FREE trial of CBS All Access! https://t.co/3P85rXLy4b https://t.co/zbWfirD9Ju
    Tweet 134: RT @instinctcbs: TONIGHT, Dr. Dylan Reinhart rewrites the book on abnormal behavior. Don't miss the premiere of #Instinct at 8/7c! https://…
    Tweet 135: If any duo knows how to rock the stage, it's @FLAGALine. The Vocal Duo Of The Year nominee will perform live at the… https://t.co/FknabB8NQp
    Tweet 136: How is your bracket looking after last night? Stream Second Round games LIVE today with a FREE trial of CBS All Acc… https://t.co/25JlIpgwog
    Tweet 137: Where better to spend #StPatricksDay than the place everybody knows your name? It’s just your luck that every singl… https://t.co/Fom5wmdENL
    Tweet 138: Stars @JakeMcDorman and Nik Dodani will join the cast in the upcoming revival of Murphy Brown coming to CBS.… https://t.co/JCAx29lo0i
    Tweet 139: RT @thegoodfight: Go behind the scenes with costume designer @DanLawsonStyle in "Behind The Style," a new weekly video series all about the…
    Tweet 140: The games have just begun! Continue to stream First Round games LIVE today with a FREE trial of CBS All Access:… https://t.co/YTGsJ48zYP
    Tweet 141: RT @TheTalkCBS: You asked, we answered! The fun never ends when the ladies #KeepTalking and answer your fan questions 🗣💬➡️ https://t.co/ie1…
    Tweet 142: RT @instinctcbs: Dr. Dylan Reinhart is lured back into the field from his life of quiet academia when a certain serial killer makes things…
    Tweet 143: Stream First Round games LIVE today starting at 12PM ET with a FREE trial of CBS All Access! https://t.co/3P85rXLy4b https://t.co/vZow3YD8cb
    Tweet 144: RT @CBSSports: It's the most wonderful time of the year. #MarchMadness https://t.co/e4c9qohqSR
    Tweet 145: Give these ladies some love! @Lauren_Alaina, @DBradbery, @carlypearce, and @RaeLynn are nominated for New Female Vo… https://t.co/IVhwURfJ3S
    Tweet 146: RT @ManWithAPlan: Hungry for more #ManWithAPlan bloopers and behind-the-scenes videos featuring cast like @matt_leblanc, @thelizasnyder, @k…
    Tweet 147: Music stars @MileyCyrus, @edsheeran, @ladygaga, and more will honor the legendary @eltonofficial and his hit songs… https://t.co/UzxARCCLnI
    Tweet 148: RT @thegoodfight: The verdict is in. The new season of #TheGoodFight is 🔥🔥🔥! Stream it now on CBS All Access: https://t.co/FkYSNSXlRb https…
    Tweet 149: RT @MadamSecretary: In less than an hour, #MadamSecretary's Keith Carradine will be taking over the @MadamSecretary Twitter page! Tweet alo…
    Tweet 150: RT @DierksBentley: Take and post a photo of the woman in your life who inspires you daily! Use the hashtag #WomanAmenACM in your post for a…
    Tweet 151: RT @MomCBS: If you missed guest star @KChenoweth in the latest episode of #Mom, not to worry! Watch now: https://t.co/RlvXoGOZ0l https://t.…
    Tweet 152: Give a round of applause to @KelseaBallerini, @MirandaLambert, @Reba, @MarenMorris, and @CarrieUnderwood, the five… https://t.co/Ncp1BTXx6N
    Tweet 153: RT @thegoodfight: Smart, sexy, and sophisticated. See what's coming this season on #TheGoodFight. https://t.co/CuKhx2G50P https://t.co/ygTI…
    Tweet 154: RT @BlueBloods_CBS: Even stand-up guys fall down sometimes. #BlueBloods is new tonight at 10/9c! https://t.co/UOlDm22wWW
    Tweet 155: Today and every day we celebrate the women in our lives who empower and inspire us. Share a story about an influent… https://t.co/9rVtqrElvT
    Tweet 156: Take and post a photo of the woman in your life who inspires you daily! Use the hashtag #WomanAmenACM in your post… https://t.co/7ShhvE48zy
    Tweet 157: RT @thegoodfight: Meticulously constructed. Soapy &amp; sexy. Intoxicating, savage television. 🔥 Here's what critics are saying about #TheGoodF…
    Tweet 158: This just in! @Jason_Aldean, @mirandalambert, @LukeBryanOnline, and many more are set to perform at the 53rd Academ… https://t.co/mfxw2VxzU4
    Tweet 159: Meet the ensemble of talented actors slated to join $1, a new mystery series coming to CBS All Access:… https://t.co/QoyYv7vxwg
    Tweet 160: Will @Jason_Aldean, @garthbrooks, @LukeBryanOnline, @ChrisStapleton, or @KeithUrban be named Entertainer Of The Yea… https://t.co/rMD8zjeX3s
    Tweet 161: RT @thegoodfight: It feels good to be back. 👠💄🔥 The season 2 premiere of #TheGoodFight is now streaming, exclusively on CBS All Access: htt…
    Tweet 162: RT @thegoodfight: Tomorrow, #TheGoodFight is back. Stream the season 2 premiere only on CBS All Access: https://t.co/tNFR8LBJO2 https://t.c…
    Tweet 163: Who are the trailblazing women in your life that inspire you? Join CBS and the ANA's #SeeHer initiative, celebratin… https://t.co/M0KqZ41Bes
    Tweet 164: Join @maria_bello, @aishatyler and @TeaLeoni in celebrating the accomplishments of women who have contributed to th… https://t.co/MefESBeFL3
    Tweet 165: In honor of Women's History Month, CBS and the Association of National Advertisers' (ANA) #SeeHer initiative will p… https://t.co/2wtYxKJVuO
    Tweet 166: RT @ZoeListerJones: Tonight’s an all new Life In Pieces and it’s directed by my ride or die @nataliaanderson!!!… https://t.co/2LPfmyLWrY
    Tweet 167: RT @MarenMorris: Hot damn! Woke up from my post-wisdom teeth haze to find out I’m up for 4 @ACMawards ! So honored, especially for the Dear…
    Tweet 168: RT @KelseaBallerini: Ohhhhh goodness. Incredible. Thank you thank you thank you. #female https://t.co/1ZTYjNfQeF
    Tweet 169: RT @KeithUrban: ACMs...... HOLY SMOKES!!!!! MAD LOVE TO U ALL THIS MORNING  FOR THESE INCREDIBLE NOMINATIONS. I’M EXTREMELY GRATEFUL!!!!!!!…
    Tweet 170: RT @ACMawards: Congratulations to this year’s #ACMawards Video of the Year nominees:
    “Black” - @DierksBentley
    “It Ain’t My Fault” - @Brothe…
    Tweet 171: RT @ACMawards: Please give a round of applause to this year’s #ACMawards Entertainer of the Year nominees: @Jason_Aldean, @GarthBrooks, @Lu…
    Tweet 172: .@ChrisStapleton, @ThomasRhett, @mirandalambert and more are all nominated for awards at Country Music's Party of t… https://t.co/Vm1vXRUDYJ
    Tweet 173: The Queen of Country, @Reba, is returning to host the 53rd #ACMawards on Sunday, April 15 at 8/7c. Here are a few o… https://t.co/Iqzz6Gql01
    Tweet 174: RT @survivorcbs: It’s time! #Survivor https://t.co/YPk6cGWrUA
    Tweet 175: RT @CBSThisMorning: TOMORROW: The nominees for the 2018 @ACMawards will be announced live by the one-and-only, @Reba! 
    
    Watch on @CBS in ou…
    Tweet 176: RT @thegoodfight: From the set design and costumes to hair and makeup, the production quality is truly next-level. Take a peek inside the u…
    Tweet 177: RT @LivinBiblically: The fun continues on Facebook! The #LivingBiblically cast is live to talk about tonight’s premiere. Tune in here: http…
    Tweet 178: RT @KevinCanWaitCBS: Can you get all the way through these #KevinCanWait bloopers without laughing?! @KevinJames,@LeahRemini and the rest o…
    Tweet 179: RT @ACMawards: That’s right! @Reba is headed to @CBSThisMorning on Thursday, March 1 to announce this year’s #ACMAwards' nominees. Tune in…
    Tweet 180: RT @ScorpionCBS: You can't hack your way to a 197 IQ, but you are well on your way with these Genius Facts from #TeamScorpion! 💻 You can be…
    Tweet 181: RT @SuperiorDonuts: You can always count on @DavidKoechner for a laugh! Did your favorite Tush moment make the list? Catch a new #SuperiorD…
    Tweet 182: RT @TheTalkCBS: TODAY: We loved them together then &amp; we love seeing them together now! Welcome back to the show @THESaraGilbert​'s good fri…
    Tweet 183: RT @thegoodfight: As foundations begin to crumble, our characters struggle to make sense of this new dystopian world. The cast teases what'…
    Tweet 184: #LivingBiblically's @linzkraft and @jrfergjr appeared on @KCBS's Facebook Live this morning, talking all about what… https://t.co/4RebcHuuMQ
    Tweet 185: RT @CBSSports: Introducing CBS Sports HQ, a New 24/7 Direct-to-Consumer Streaming Network for Sports News, Highlights, &amp; Analysis.
    
    Stream…
    Tweet 186: RT @CBSBigBrother: It’s down to the final 5 celebrity Houseguests, and anyone could take home the grand prize! Tune in NOW to watch the #BB…
    Tweet 187: RT @startrekcbs: Binge the entire first season of #StarTrekDiscovery. All episodes now streaming exclusively on CBS All Access: https://t.c…
    Tweet 188: RT @thegoodfight: #TheGoodFight returns in 1 week. Season 2 premieres Sunday, March 4. https://t.co/nomCao1GWp https://t.co/BOn6bOe9Tb
    Tweet 189: RT @thegoodfight: This is our new favorite thing. Christine Baranski debuted #TheGoodFight the Musical on @colbertlateshow last night! 🎵🎤…
    Tweet 190: RT @LivinBiblically: Confession time: have YOU ever hit the "close door" button in an elevator while somebody was approaching? The cast of…
    Tweet 191: RT @CBSEyeSpeak: Mark your calendars! #CBSEyeSpeak kicks off March 14 with The EYE Speak Summit. Follow our page for more details! https://…
    Tweet 192: RT @CBSEyeSpeak: Proud to announce a new CBS initiative, promoting female empowerment and developing the next generation of leaders through…
    Tweet 193: RT @LivinBiblically: When you're living by the Bible, it's good to have a priest and a rabbi on call (provided they answer their phones, th…
    Tweet 194: RT @thegoodfight: Chicago lawyers are being hunted and the world is going insane. 
    
    The new season of #TheGoodFight premieres Sunday, March…
    Tweet 195: Ready for some larger than life competition? This new series from @MarkBurnettTV will premiere in summer 2018.… https://t.co/gDXHLdIJ5v
    Tweet 196: With tournament dreams on the line, make sure to stream these college basketball matchups on CBS All Access:… https://t.co/SGkYUZrQWB
    Tweet 197: RT @LivinBiblically: While Chip's sticking to the Bible's original rules, the cast of #LivingBiblically has given them a more modern makeov…
    Tweet 198: Casting News! Peter Mark Kendall, Michael Gaston, Greg Wise, Rade Šerbedžija, Zack Pearlman, and Keye Chen join the… https://t.co/GFob2KrD8H
    Tweet 199: RT @BullCBS: The verdict is in...#Bull is the perfect Valentine! ❤️ Happy #ValentinesDay! https://t.co/poEejI4AnC
    Tweet 200: RT @NoActivityCBS: Car 27 reporting: Season 2 of #NoActivity coming soon!
    
    Binge season one now on CBS All Access: https://t.co/yvxoQMeyhN…
    Tweet 201: Arkansas is trying to make Big Pharma pay for the opioid crisis, accusing drug manufacturers in a new lawsuit of in… https://t.co/RGJDaMUOeY
    Tweet 202: "She's only apologizing after a third of her advertisers pulled out," says Parkland survivor David Hogg, responding… https://t.co/UNxGLzeuSk
    Tweet 203: The Trump administration will require visa applicants to submit five years of social media history https://t.co/M6qbYd6Uv1
    Tweet 204: Attorney Gloria Allred has withdrawn from representing Summer Zervos in her defamation suit against President Trump… https://t.co/B5nHFIxFv8
    Tweet 205: To critics who say she should "go away" after losing to Trump, Hillary Clinton says, "they never said that to any m… https://t.co/3RhjbE0NQh
    Tweet 206: SpaceX launched another rocket on Friday, and this time it tried to land the $6 million nose cone into a giant seab… https://t.co/NMBsh3OQuX
    Tweet 207: This Chinese space lab could plummet back to earth as early as Saturday https://t.co/ZSc6omLCy9 https://t.co/jgjCZ5IqQJ
    Tweet 208: An off-duty police officer died Thursday in Kentucky after a man impersonating an officer shot him, police say… https://t.co/20vfVWu5D6
    Tweet 209: If you stand inside the world's quietest room for long enough, you start to hear your heartbeat. Then you lose your… https://t.co/eBQKi8TPlh
    Tweet 210: This man spent more years behind bars than any other wrongfully imprisoned person in America https://t.co/L9pLu1Q6nL https://t.co/kf3WW43anM
    Tweet 211: Mark Zuckerberg has disavowed an internal memo written by a top Facebook executive in 2016 that argued growth shoul… https://t.co/rt6Orqn7NR
    Tweet 212: A Connecticut Democratic congresswoman is apologizing after she kept a top aide on her payroll for several months d… https://t.co/rChtdK8tSk
    Tweet 213: Russia released video footage Friday of a test launch of its new "Satan 2" intercontinental ballistic missile… https://t.co/Ac0UqHuxfd
    Tweet 214: JUST IN: Noor Salman, the widow of the Pulse nightclub gunman, was found not guilty of charges in connection with h… https://t.co/LeDjGXUeP1
    Tweet 215: At least eight Palestinians were killed and more than 1,000 injured in confrontations with Israeli security forces… https://t.co/dnAn5jY4N3
    Tweet 216: JUST IN: A US service member was killed in an improvised explosive device attack in Syria on Thursday, an official… https://t.co/zfLUB6dPzd
    Tweet 217: Coffee may come with a cancer warning label in California https://t.co/x4VA27Gloy https://t.co/ycwyKtFjd7
    Tweet 218: New footage shows Kim Jong Un holding court inside his armored train https://t.co/jirt7bIM1q https://t.co/Izv0nlm0s7
    Tweet 219: SpaceX is launching one of its Falcon 9 rockets and is expected to make an experimental attempt to guide the rocket… https://t.co/CCpVW3Voo7
    Tweet 220: Trump and "Roseanne" are making a conservative case for representation in media | Analysis by Hunter Schwarz… https://t.co/asAl350Eta
    Tweet 221: Two Baton Rouge police officers involved in the 2016 shooting death of Alton Sterling are expected to learn today w… https://t.co/h6e0RLJGap
    Tweet 222: China says a gang used drones to smuggle almost $80 million worth of smartphones https://t.co/689tFCKGZ6
    Tweet 223: "She's only apologizing after a third of her advertisers pulled out," says Parkland survivor David Hogg, responding… https://t.co/XZ7G5mGkCx
    Tweet 224: The amount of fan mail the Parkland shooter is receiving is unreal https://t.co/gVaR06QrQZ
    Tweet 225: Venezuelans lack Communion wafers this Easter, so a Colombian church stepped in to help https://t.co/Ch7m3zb9wi https://t.co/wiNo0G3OTw
    Tweet 226: A Russian hacker suspected of stealing 117 million LinkedIn passwords in 2012 has been extradited to the US after a… https://t.co/PdXRQ3g0wO
    Tweet 227: "What kind of dumbass colleges don't want you?" Alisyn Camerota asks Parkland survivor David Hogg, who says he was… https://t.co/VkwsfeymQ2
    Tweet 228: A company donated $29 million in cryptocurrency to cover every single teacher request on a crowdfunding site… https://t.co/tzwcncFirF
    Tweet 229: RT @NewDay: Alisyn Camerota: What kind of dumbass colleges don't want you?
    
    Parkland survivor and gun control activist David Hogg: They rej…
    Tweet 230: Russia's RT television network will go dark in Washington DC https://t.co/OyOaU1HHx2 https://t.co/UL5etAuOmE
    Tweet 231: Days after five members of the same family were killed when their SUV went over a cliff, investigators are still lo… https://t.co/HfUoN0Assu
    Tweet 232: Austin's police chief now says he would labeled the bomber who killed two people and injured several others a "dome… https://t.co/mdGrndhvr0
    Tweet 233: Johan van Hulst, a former Dutch senator and teacher who saved hundreds of Jewish children during the Holocaust, die… https://t.co/ofuqyZ2AU1
    Tweet 234: "We're knocking the hell out of ISIS. We'll be coming out of Syria like very soon. Let the other people take care o… https://t.co/0GwN6D2VAj
    Tweet 235: Do you dream of escaping to the country and running your own bookstore? There's an Airbnb for that… https://t.co/wTpI0ceCRz
    Tweet 236: If you want to stop Putin, follow the money, say CNN security analysts https://t.co/sF8HghDAkm
    Tweet 237: She chronicled her life on Instagram: her dreams, a breakup, her recovery. But it was all fake.… https://t.co/LFaO9osD6S
    Tweet 238: The best travel photos of 2018 (so far) https://t.co/ulIqOBeMUo https://t.co/b22ZRcPTX8
    Tweet 239: Bitcoin's price has slumped roughly 50% since the start of the year https://t.co/pijJ60Es4d https://t.co/4ylxLFEYel
    Tweet 240: This Chinese space lab could plummet back to earth as early as Saturday https://t.co/Tfn0lGXXXp https://t.co/o7jyh0A1f1
    Tweet 241: Tourists will soon be limited to three hours at the Taj Mahal https://t.co/TBmpRPJ1Uh https://t.co/kgoXmxhq6x
    Tweet 242: Would you pay more for your shopping to tackle plastic pollution?
    
    People in the UK could soon have to pay a deposi… https://t.co/c2Gyju6jbK
    Tweet 243: Dining out frequently is known to increase one's intake of unhealthy sugars and fats. But a new study suggests that… https://t.co/JyuecCrFsI
    Tweet 244: Here are some of iOS 11.3's features you should know about:
    - You can turn off the controversial iPhone-slowing fea… https://t.co/mAqzVGNDoS
    Tweet 245: If you stand in it for long enough, you start to hear your heartbeat. Then you lose your balance, because the absol… https://t.co/XWTyOK5bbq
    Tweet 246: A longtime public defender said he's never seen a defendant get so many letters https://t.co/WzH2qXFSUH
    Tweet 247: There are visible, external signs that can indicate if something is wrong with your heart -- check your fingers, ea… https://t.co/a40x3puz8L
    Tweet 248: Human rights lawyer Amal Clooney will represent two Reuters journalists who have been jailed in Myanmar and accused… https://t.co/zSi0goEyKl
    Tweet 249: The Trump administration will require immigrants to submit five years of social media history https://t.co/i8HJgnvpIF
    Tweet 250: K-pop girl band Red Velvet are among the South Korean stars travelling to North Korea this weekend as they get read… https://t.co/dDzBm2lCDd
    Tweet 251: A Chinese space lab could plummet back to earth as early as Saturday, authorities say, in a fiery end to one of the… https://t.co/O1PI1M4roe
    Tweet 252: "I heard you're actually the devil incarnate and I wanted to meet you," Defense Secretary James Mattis joked during… https://t.co/Tk9IcQi2ei
    Tweet 253: This Easter, don't let politics define forgiveness | By Bob Vander Plaats via @CNNOpinion https://t.co/iDUc7yxvdK https://t.co/WLbwHzH2BS
    Tweet 254: Former French President Nicolas Sarkozy will face a trial on charges of corruption and influence peddling, a source… https://t.co/l3mfwOMuLP
    Tweet 255: Spotify is about to go public — and one analyst thinks it could be worth $43.5 billion https://t.co/ABUdIBQics https://t.co/UuMrHQrp61
    Tweet 256: Hillary Clinton hasn't hit the campaign trail yet on behalf of Democrats running in 2018, but that isn't stopping R… https://t.co/RBGnalvhZU
    Tweet 257: An off-duty police officer died on Thursday in southwestern Kentucky after a man impersonating an officer shot him,… https://t.co/9n2qeQi6Ms
    Tweet 258: The crumbling colonial-era churches of Pakistan https://t.co/hKlARundSu via @CNNStyle https://t.co/eUQK2f6QIM
    Tweet 259: Family's SUV goes over a cliff, leaving authorities seeking 3 children and clues to what happened… https://t.co/bBxOVEdCgf
    Tweet 260: A Chinese space lab could plummet back to earth as early as Saturday, authorities say, in a fiery end to one of the… https://t.co/H4L5vPk1E8
    Tweet 261: You've heard of SpaceX landing and reusing rockets. But safely recapturing the $6 million nose cone that sits at th… https://t.co/CHfIj7PRRb
    Tweet 262: Japan revealed it was seeking a summit with North Korean leader Kim Jong Un, as some in Tokyo expressed concern the… https://t.co/oSIVX046l3
    Tweet 263: One of India's most famous landmarks, the Taj Mahal, is planning to place a three-hour cap on visits to avoid overc… https://t.co/xXG9UwEd6z
    Tweet 264: So-called stumbling stones -- memorials to victims of the Holocaust -- bring history to life for a new generation o… https://t.co/uu4ntF0PU8
    Tweet 265: Russia will expel 60 US diplomats and close the US Consulate in St. Petersburg, Foreign Minister Sergey Lavrov has… https://t.co/IkuwlwA51b
    Tweet 266: If we want to stop Putin, we need to go after his sources of money | By Josh Campbell and Robert Baer via… https://t.co/21HuzQ8QBM
    Tweet 267: A legal defense fund site has been set up for former FBI deputy director Andrew McCabe https://t.co/CcT2hvpHTE https://t.co/SJrGuIGQWc
    Tweet 268: A legal defense fund site has been set up for former FBI deputy director Andrew McCabe https://t.co/79lGlgOORC https://t.co/oXrh0lCSST
    Tweet 269: This hockey player breastfeeds her baby during game breaks -- like a total mom boss https://t.co/oGfHwRTLOu https://t.co/UxOseVENRk
    Tweet 270: A company donated $29 million in cryptocurrency to cover every single teacher request on a crowdfunding site https://t.co/hzJ2d9Vpya
    Tweet 271: "I heard you're actually the devil incarnate and I wanted to meet you," Defense Secretary James Mattis joked during… https://t.co/CgIyVne1y4
    Tweet 272: Two commercial pilots flying over the Arizona desert claim they saw an unidentified flying object pass overhead, ac… https://t.co/OyWw74DsEP
    Tweet 273: More than 200 retired US diplomats are sounding the alarm about diplomacy under Trump, urging lawmakers to ensure t… https://t.co/2u1BabDKLi
    Tweet 274: This curious cheetah hopped into a car on a safari trip. Thankfully, everyone remained calm, and no one was hurt… https://t.co/oK2WrcRLFs
    Tweet 275: Would you ride in this "Roseanne" car? A New York City subway train car has been transformed to resemble Roseanne's… https://t.co/uSm5W59cfP
    Tweet 276: The EPA circulated new talking points that downplay the role of human activity in climate change, instructing staff… https://t.co/QEjx1eqKb2
    Tweet 277: Spotify is about to go public — and one analyst thinks it could be worth $43.5 billion https://t.co/YFGqBucxsX https://t.co/oxHsXGdhVh
    Tweet 278: A London orchestra's brass section was so loud a violist says it ruined his hearing. A judge agreed.… https://t.co/6yNRrWsEdM
    Tweet 279: The Trump administration plans to require nearly all visa applicants to the US to submit five years of social media… https://t.co/utAwtGucsr
    Tweet 280: Days after five members of the same family were killed when their SUV went over a cliff, investigators are still lo… https://t.co/TxwiLmAomF
    Tweet 281: Human rights lawyer Amal Clooney has agreed to represent two Reuters journalists who have been jailed in Myanmar an… https://t.co/8Bo1vTvhYY
    Tweet 282: Draylen Mason had just been admitted to a prestigious music school, but the 17-year-old died in the Austin bombings… https://t.co/kPiOxoRerr
    Tweet 283: Sears CEO Eddie Lampert got a  24% raise last year, despite the company's financial troubles https://t.co/NTag87tUEB
    Tweet 284: Howard University students protested following news that six university employees were fired for "double-dipping" f… https://t.co/j5wSHApjHt
    Tweet 285: ExxonMobil won't be able to stop state investigations into whether it misled investors and the public about its kno… https://t.co/30xneAmd3x
    Tweet 286: This Easter, don't let politics define forgiveness | By Bob Vander Plaats via @CNNOpinion https://t.co/tDbOCBMjEC https://t.co/fjGkv5PQTY
    Tweet 287: Johan van Hulst, a former Dutch senator and teacher who saved hundreds of Jewish children during the Holocaust, die… https://t.co/HPHHxgPK7p
    Tweet 288: Russia's RT television network will go dark in Washington DC https://t.co/TA4SmzpqoC https://t.co/yFyj9jXPd0
    Tweet 289: Judge Stephen Reinhardt, a liberal federal appeals court judge who was part of a panel that rejected California's P… https://t.co/OiL15dux5h
    Tweet 290: Sarah Jessica Parker has endorsed her "Sex and the City" costar Cynthia Nixon for New York governor… https://t.co/kOxwnvg8EP
    Tweet 291: A court once again orders a new trial for Adnan Syed, the subject of a "Serial" podcast https://t.co/f9awk8fpoI https://t.co/nGhyWFsoi3
    Tweet 292: RT @AC360: Fired VA secretary says his replacement is a "person who is honorable and cares about our veterans" and pledges to help him thro…
    Tweet 293: The "Sharknado" movies will end after a sixth installment this summer https://t.co/LFghVNqMG3 https://t.co/xitTwRA2yp
    Tweet 294: India is building a city from scratch to attract foreign investors https://t.co/0w7l1CpKA7 https://t.co/lrjXCDlua1
    Tweet 295: China's rap scene has been frustrated by a crackdown.
    
    Last year, the genre was having something of a heyday, but t… https://t.co/Kl8qQWAtXj
    Tweet 296: There are visible, external signs that can indicate if something is wrong with your heart -- check your fingers, ea… https://t.co/wCXF1zrUlX
    Tweet 297: The Trump administration will no longer seek to automatically release pregnant immigrants from detention -- a move… https://t.co/QjtCGmRRBM
    Tweet 298: One fought her abuser every night, another stood up to her groping boss. Meet the women driven by the #MeToo moveme… https://t.co/FD5flQof8n
    Tweet 299: Abortion funds band together to sue their cyberattackers https://t.co/bkEMLRCLjB https://t.co/mPvroEKFqZ
    Tweet 300: Fox News host Laura Ingraham apologized for a widely derided tweet in which she mocked Parkland survivor David Hogg… https://t.co/mZ3grjS0wW
    Tweet 301: @glenngillen I can tell a lot from about / careers pages and how a business conducts themselves. :)
    Tweet 302: RT @jessamyn: NYTimes releases Diversity and Inclusion Report. With graphs that have differing Y axes that subtly give the wrong impression…
    Tweet 303: @mjmichellekim also, thanks to you I’ve just discovered you can pitch pieces to Quartz 🙇🏻‍♀️
    Tweet 304: “To achieve lasting change, you have to focus on something bigger than what you can measure in the short-term.” 
    
    Y… https://t.co/4b5womubML
    Tweet 305: I’m one of those people who create a spreadsheet for all the organisations I’m interviewing with, gather their dive… https://t.co/2RsVggo5yI
    Tweet 306: @noopkat I subscribe to this newsletter
    Tweet 307: @kissane Sorry for being unclear! I think citation is great, I often found resources I haven’t seen before, plus al… https://t.co/jcD3ZGYM7m
    Tweet 308: @kissane So the approach of just hot linking to a website out there without considering how people could report, wh… https://t.co/MrFZRDGLY6
    Tweet 309: Just to clarify: linking to https://t.co/a6Ht7BpUMZ or other examples out there doesn’t constitute an inclusion or… https://t.co/tktVbndmgi
    Tweet 310: @kissane Yeah, that is good! I meant the events that link directly to public Code of Conducts that have no contact details, etc.
    Tweet 311: If a Code of Conduct of an event links to one of the community-hosted, external CoCs, what it tells me is that the… https://t.co/K27uhTjXbg
    Tweet 312: AirBnB giving out user data to the Chinese government, Facebook discriminating through ads and leaking user informa… https://t.co/lAHF6sJuKG
    Tweet 313: RT @sw: This is so exciting, @triketora lands in the Southern Hemisphere! 🎉🐧🐳 https://t.co/I5czjV84uE
    Tweet 314: Thank you for the shout-out to inclusion at CSSConf AU @kylietimpani 🙇🏻‍♀️
    
    Make sure you read the entire interview… https://t.co/S21RBI6s2y
    Tweet 315: RT @Future_Females: Happy #MuslimWomensMonth ! Today we're watching one of our favourite advocates for tolerance and diversity, @yassmin_a…
    Tweet 316: I love how sometimes people advise underrepresented groups to take on opportunities that require significant salary… https://t.co/eYww0rgKAE
    Tweet 317: RT @catehstn: Discovering the five causes of burnout that are not overwork informed this approach. It’s such a useful way to understand bur…
    Tweet 318: A Tim Tam a day keeps the doctor away.
    
    —ancient Australian proverb
    Tweet 319: I’ve been watching Big Little Lies lately, and the portrayal of the complexity of abusive relationships is both exc… https://t.co/qMNxveax5O
    Tweet 320: @RedRoxProjects Thank you! 💐
    Tweet 321: @OphelieLechat 💓
    Tweet 322: This, a thousand times for job interviews or ”I just want to catch up and talk about things”. https://t.co/5AA2d4vkwd
    Tweet 323: @SaskiaFairfull North Melbourne Meat Market.
    Tweet 324: @SaskiaFairfull Thank you!
    Tweet 325: This incredible venue plus my art direction skills. I’m pretty proud of myself.
    
    💖 https://t.co/st8XZOoc4y
    Tweet 326: Have you ever seen a conference releasing a diversity report, even if it’s only in the spectrum of gender?
    
    I haven… https://t.co/e819on1BUh
    Tweet 327: @darngooood I’m perpetually inspired by your clothing style. ✨😭😍
    Tweet 328: @mspowahs ugh damn Facebook, the root of evil
    Tweet 329: Company: Diversity and inclusion are really important to us!
    
    Company has:
    
    → Zero women
    → Women only in support, m… https://t.co/MZmNK67D6a
    Tweet 330: RT @zeynep: Show me your budget and business model, and I’ll know your actual priorities. The rest is mostly PR. https://t.co/hy3XyoW3Lt
    Tweet 331: @jennwrites the voices are coming from inside the house
    Tweet 332: @lara_hogan Congratulations! You both look fantastic. 💝
    Tweet 333: RT @menajew: There’s a common misconception that you can’t be Disabled and happy. That you can’t express joy, and if you do, you must not b…
    Tweet 334: Meanwhile in Poland, where the government is still trying to push an abortion ban and curb women’s rights. https://t.co/zTmJo9P2gy
    Tweet 335: A casual reminder that you don’t need to be an entrepreneur, founder, C-level exec or a manager for your work to ma… https://t.co/jcxAagr1cL
    Tweet 336: @kouky @benschwarz It’s for me, obviously
    Tweet 337: Seriously, @wnotw? Alcohol is the top perk you can think of? https://t.co/itaV4XwmOZ
    Tweet 338: RT @sarahcuda: this is well put. https://t.co/5sDOcqX29b
    Tweet 339: I need this pin immediately. https://t.co/EHTEFefzop
    Tweet 340: @dgoodlad might be of your interest https://t.co/pS2X5ExrLs
    Tweet 341: Less mentoring, more empowerment.
    Tweet 342: My life in a nutshell (plus a few great strategies on not talking about “being a woman in tech”).
    
    👏🏻 @vaurorapub!
    
    https://t.co/aTlTP0F8jI
    Tweet 343: Do yourself a favour and read this fantastic piece by @sw and @vaurorapub on getting free from toxic tech culture.
    
    https://t.co/CzPs8FaBcL
    Tweet 344: After running a 4-day conference and I’m now back on calling out exclusion and toxic tech culture.
    
    💁🏻‍♀️ 👋🏻
    Tweet 345: @benschwarz @adactio Also coffee at Karma, sweets at Massolit Bakery, wine at Charlotte, beer at Multi Qlti :)
    Tweet 346: RT @bubsolow: starting the job hunt ☞ would appreciate any junior developer roles / feedback on my resume https://t.co/PiSPoASsEf
    Tweet 347: @jennschiffer @laras126 @gr2m @jimthoburn @rouzbeh84 Yep, I just wanted to clarify why I was asking :)
    Tweet 348: @jennschiffer @laras126 @gr2m @jimthoburn @rouzbeh84 I understand that, however I’d like to point out that there ar… https://t.co/Bn3xxN1M2S
    Tweet 349: My event organizing performance review lies in community impressions. 🙏🏻 https://t.co/4mzFEwwzKR
    Tweet 350: @laras126 @gr2m @jimthoburn @rouzbeh84 @jennschiffer Hmmmm... did that go through the rules for running a CSSConf? https://t.co/KaFgIwo4Rd
    Tweet 351: I wish all of the brands capitalizing on the feminist movement at least had some baseline of inclusion.
    
    Make it in… https://t.co/yK3F8MMBP0
    Tweet 352: Looking to read some new, good product and design books. Any recommendations for publications within the last 3 years? 🙏
    Tweet 353: @tsunamino omg, so excellent!
    Tweet 354: @sch Sure! Thank you for being open to chatting. Emailed. :)
    Tweet 355: @sch I’m in Melbourne, which has 5 hours overlap with SF :)
    Tweet 356: @sch hey! I’m assuming this isn’t a remote job?
    Tweet 357: @jasonfried I always really enjoyed 37signals’ books, but I can’t unsee the ableism in the title...
    Tweet 358: I’d love to have more time to build up my professional portfolio but it’s hard to focus when there’s no job securit… https://t.co/QOnLi6v91M
    Tweet 359: As much as I love connecting with fellow women at tech events, it always breaks my heart so much that half of the c… https://t.co/DlZtj3F2AQ
    Tweet 360: @mapotato I’m so sorry about this, Theresa. We will try to find out the name of the offender. There will be consequences.
    Tweet 361: Inclusion sections on careers pages mean nothing if you can’t reply to all candidates no matter the outcome and giv… https://t.co/JrBPxHYF7m
    Tweet 362: @almonk @jgwhite @itchymutt Interesting. I’d be keen to chat in that case :)
    Tweet 363: RT @duyenho: Hats off to @Fox for being the real deal 💪 Thank you Karolina and @benschwarz for the last ever #jsconfau18 https://t.co/8oRoy…
    Tweet 364: RT @not_sherry_wine: Shoutout to @fox and @benschwarz for the fantastic #JSConfAU18. The opportunity program is one of the best gift I’ve b…
    Tweet 365: RT @the_patima: 📢✨ I have something to tell you all...
    
    I'm going to Berlin in June to speak at @jsconfeu!!!! 💖
    
    🌟 It's my 1st ever trip to…
    Tweet 366: @itchymutt @heroku SF-based?
    Tweet 367: RT @susanthesquark: I've said it a million times and I'll keep shouting it from the rooftops: a company's diversity and inclusion efforts m…
    Tweet 368: Apparently someone said that they’re only afraid of meeting two people; @dhh and myself.
    
    I will take this as a compliment.
    Tweet 369: That’s a wrap, everyone. CSSConf and JSConf Australia are now a thing of a past.
    
    Hope we left you inspired and with new friends.
    
    🙌
    Tweet 370: @robodana Thank you for your kind words, Dana! I’m glad you’ve enjoyed it
    Tweet 371: @kriesse @benschwarz Texting from bed, yes 🌸
    Tweet 372: @sentience Ah, interesting! Great photos 👏🏻
    Tweet 373: @sentience what camera and lenses are you using? Fuji or Leica? The quality is quite great.
    Tweet 374: RT @JSConfAU: Did you enjoy the live stream today? Great, because we’re doing it again today.
    
    Tune in for awesome tech talks. 🙏❤️ #jsconfa…
    Tweet 375: I have so many subtweets about people misbehaving in subtle ways at conferences I could probably write a book by now. 
    
    Last day to go!
    Tweet 376: RT @fox: 🚨👋🏻 Hi everyone! I’m searching for new opportunities. 
    
    I’m looking for multidisciplinary product, front-end and leadership roles.…
    Tweet 377: @RedRoxProjects Thank you! 🌺
    Tweet 378: @agisilaosts thank you! 🌺
    Tweet 379: We’re live streaming @jsconfau talks all day today and tomorrow. Starting in 30 minutes!
    
    https://t.co/ttIzoV4Cjz
    Tweet 380: RT @frameshiftllc: New angle for change in VC: founders refusing to take funding from all-male all-white firms https://t.co/AMi8UJbTGO
    Tweet 381: @sentience @benschwarz @glenmaddern @xzyfer @jordanlewiz I like how everyone is wearing black or tan pants.
    Tweet 382: @kriesse @benschwarz ❤️❤️❤️
    
    I’ll have a celebration of lying face down in a few days.
    Tweet 383: @butwhoiskat thank you!
    Tweet 384: Here it goes. Last 8 months of constant work will happen within the next 3 days.
    
    https://t.co/c48sQqdnXv
    Tweet 385: RT @cssconfau: Guess what?
    
    Even if you aren’t here, you can watch all the talks LIVE! 😱
    
    Tune into the live stream here. #cssconfau18 
    
    ht…
    Tweet 386: @amyngyn hahaha. I actually never been, but I’ve heard the bar is great. :)
    Tweet 387: @amyngyn Loui Bar?
    Tweet 388: RT @LJKenward: Hey friends! 👋 Who's hiring Junior Devs at the moment? I have some awesome people from the @juniordev_io Community currently…
    Tweet 389: Don’t forget about the Community Social today! EVERYONE IS WELCOME (even if you don’t hold a CSSConf or JSConf tick… https://t.co/6c79TcKFCi
    Tweet 390: Toxic tech industry creates a fake vision of what each of us (especially minorities) have to be and achieve to be ”… https://t.co/yfXZ5nxCJ7
    Tweet 391: @amyngyn I never introduce myself. The focus is on content, not myself. Also I don’t feel like I need to justify my cred. :)
    Tweet 392: Today I got kissed by a dingo. 💁🏻‍♀️ https://t.co/FDQsVw2anl
    Tweet 393: @Sareh88 Thank you, Sareh! That’s very kind. ❤️
    Tweet 394: @meelijane https://t.co/Y5wM3nCdsH in Northcote. I’ve tested many and this one is orders of magnitude better than everywhere else. :)
    Tweet 395: One of many reasons why I love my yoga studio so much is how meditative the practice is and how all the instructors… https://t.co/VXfSqdo6bk
    Tweet 396: @IvanaMcConnell I can only help ruin your bank account further, sorry. 😂
    Tweet 397: RT @slamup: people love to say
    
         no child is born 
         with hate in their heart
    
    which is all very
    romantic
    
    but from the moment
    a bla…
    Tweet 398: @evanderkoogh Nope, we are at full capacity of the venue. :)
    Tweet 399: RT @cssconfau: Come and celebrate with us at pre CSSConf and JSConf AU community social!
    
    📅 Monday, March 19, 6pm onwards
    📍Stomping Ground…
    Tweet 400: @noahmp Heh, dang :) worth asking nonetheless.
    Tweet 401: RT @nytimesarts: Kate Mara on her new movie “Chappaquiddick”: “Like a lot of people, I’m fascinated with the Kennedys and their history and…
    Tweet 402: RT @NYTStyles: The Wing is under investigation for discrimination, but honestly ... that's only gotten it more press. https://t.co/uHGB7sVY…
    Tweet 403: In college basketball, Catholic schools have long punched well above their weight. The reasons stretch back a centu… https://t.co/wJkwuvDZRS
    Tweet 404: RT @dgelles: Cash may be king, but it's no longer essential. 
    
    I've gone cashless. My essay here: 
    https://t.co/laHrQD9j1F
    Tweet 405: The midterm elections will determine the political script for the rest of President Trump’s first term. And Pennsyl… https://t.co/ZpZGFxjunA
    Tweet 406: Puerto Ricans are trickling back to the island. They must come to terms with a Puerto Rico that is still crippled,… https://t.co/Wml9x3rYyw
    Tweet 407: Hope Hicks has left the building. Those who remain are wondering what happens now. https://t.co/ds3NJlS4i8
    Tweet 408: In Opinion
    
    Alex Wagner writes: "Racially speaking, the United States is 0% Hispanic. This is confusing — especiall… https://t.co/kzJaxIOguf
    Tweet 409: A California judge’s ruling would require cancer warning labels on coffee. The coffee industry is considering how t… https://t.co/1GJ1f9b3Mj
    Tweet 410: "That’s how I ended up losing my virginity on a fourth date with a middle-school teacher that I didn’t even particu… https://t.co/Dk0vk6HRT2
    Tweet 411: Breaking News: What was billed as a peaceful protest along Gaza's border with Israel turned bloody. The Israeli mil… https://t.co/RVW9kEuN82
    Tweet 412: There’s been talk of bankruptcy swirling around Gibson, the Nashville-based guitar company. What happened? https://t.co/cOVKNf2Wbm
    Tweet 413: A Russian man accused of hacking 3 American tech companies in 2012 has been extradited to the U.S. https://t.co/MnltMkO1Oe
    Tweet 414: RT @jodikantor: Six months ago, @ronlieber wrote about teachers and firefighters being seriously screwed by a government student loan progr…
    Tweet 415: Another week, another news quiz. Tell us how you did. https://t.co/buvo5zeiKz
    Tweet 416: President Trump insisted he will never again sign a huge catchall spending bill. Given the state of Congress,… https://t.co/Ov68bYiDwD
    Tweet 417: Exxon sued to try to block investigations into its research and public statements about climate change. A judge dis… https://t.co/uFOMCdotBj
    Tweet 418: Breaking News: Noor Salman, the widow of the Pulse nightclub shooter, was found not guilty of helping her husband c… https://t.co/O1tNykTV7B
    Tweet 419: By tying a trade deal with South Korea to progress in denuclearizing North Korea, President Trump is showing how li… https://t.co/IfnzE5hDEO
    Tweet 420: RT @UpshotNYT: Does more immigration mean more crime? In collaboration with the @MarshallProj we look at the data behind this widely held i…
    Tweet 421: Why do cracking knuckles make that sound? You might need a calculator to explain it. https://t.co/YpjnR3SXN2
    Tweet 422: 5 times David Pecker and The National Enquirer have defended or championed President Trump https://t.co/1BQX7SYOm5
    Tweet 423: RT @PatriciaMazzei: BREAKING: Jury in Salman trial has reached a verdict. It will be announced in court in about 30 minutes.
    Tweet 424: RT @nytimesworld: “We’ll probably hold that deal up for a little while, see how it all plays out.”—President Trump on his recently announce…
    Tweet 425: RT @nhannahjones: I went on The Daily with @mikiebarb today to discuss Linda Brown and the continuing fight to fulfill the promise of landm…
    Tweet 426: “I want to die over there, not here,” said Enrique, lying on the hotel bed, a scar from one of his many surgeries v… https://t.co/KNpdOHhDJW
    Tweet 427: RT @nytimesworld: Couples in Denmark can obtain a divorce in less than a week. The government is trying to make breaking up a little a hard…
    Tweet 428: The tech industry was once one of President Trump's most vocal opponents.  Now it has increasingly found common gro… https://t.co/TAjx9fZATB
    Tweet 429: Thousands of asylum seekers remain stranded on the Greek island of Lesbos, unwilling to go back to the countries th… https://t.co/LrxHQuWlTN
    Tweet 430: RT @peterbakernyt: Trump's attacks on Amazon are often set off by his anger at stories in the @washingtonpost, also owned by Bezos, associa…
    Tweet 431: Morning Briefing: Here's what you need to know to start your day https://t.co/aNoHwClgkV
    Tweet 432: The "Mean Girls" musical comes 14 years after the movie. And Gretchen is still trying to make "fetch" happen. https://t.co/GEfZu6meAR
    Tweet 433: RT @motokorich: No “Gangnam Style” for North Korea, but plenty of Kpop will be playing In Pyongyang. (In which I get the fun reporting assi…
    Tweet 434: The death of Anthony Weber, a 16-year-old killed in a police shooting, is a sign to many in Los Angeles of how much… https://t.co/KPSosHPLdb
    Tweet 435: At least 5 Palestinians were reported killed in clashes with Israeli soldiers in Gaza https://t.co/Ug9FXsdXRW
    Tweet 436: Linda Brown, who was at the center of Brown v. Board of Education , died on Sunday. Who was she, and what has chang… https://t.co/ePfthSQoxp
    Tweet 437: Laura Ingraham, a Fox News host, apologized for taunting a Parkland shooting survivor after losing advertisers https://t.co/HwwUsjEkMI
    Tweet 438: In Opinion,
    
    @rgay writes: "No amount of mental gymnastics can make what Roseanne Barr has said and done in recent… https://t.co/E5pGePnQ9B
    Tweet 439: The story of one White House dinner shows how the National Enquirer's publisher has used President Trump's friendsh… https://t.co/pg0FPgDFLk
    Tweet 440: Clashes on Gaza’s Border With Israel as Protests Start https://t.co/cVE2wFRWvk
    Tweet 441: Hope Hicks has left the building. Those left behind are wondering what happens now. https://t.co/LOiA3Ckq6n
    Tweet 442: Opinion: When the Dream of Economic Justice Died https://t.co/KWw81L0MVi
    Tweet 443: The U.N. secretary general called climate change “the most systemic threat to humankind” https://t.co/bc0Bo7Ztww
    Tweet 444: Gym.
    Tan.
    Laundry.
    Check on the kids. 
    https://t.co/7Yx2f0vUGd
    Tweet 445: By the Book: Ernest Cline: By the Book https://t.co/i5c0lu38kG
    Tweet 446: Fact Check: The Facts Behind Trump’s Tweet on Amazon, Taxes and the Postal Service https://t.co/eDFR9JldgC
    Tweet 447: How to Avoid a Renovation Nightmare https://t.co/5q8Y0OBJWh
    Tweet 448: In 2014, one of the Hart family's six adopted children drew widespread attention when he was photographed hugging a… https://t.co/QPSk6dactX
    Tweet 449: Here’s how @bxchen scrubbed his Twitter and Facebook timelines — and what 
    he learned https://t.co/hlSw7hAcaO
    Tweet 450: To Raise Resilient Kids, Be a Resilient Parent https://t.co/1NmwYvZWhN
    Tweet 451: The dilemma confronts government officials frequently. Digital extortionists have hijacked their computer systems a… https://t.co/IC8xB5WX2w
    Tweet 452: There remains a strong correlation between marriage and economic class in the American higher education system https://t.co/0uV50eIdmI
    Tweet 453: There remains a strong correlation between marriage and economic class in the American higher education system https://t.co/uTEZvVuNCT
    Tweet 454: The NBA is far more willing to address social issues than other leagues. But the Sacramento Kings' partnership with… https://t.co/QWR6Rd2vVa
    Tweet 455: RT @nickconfessore: David Pecker, Jared Kushner, and an advisor to the Saudi crown prince walk into the Oval. You can’t make it up—but you…
    Tweet 456: Anbang was seized by the Chinese government. It's still offering deals to investors with promises of safety. https://t.co/1TD9RzC7Ep
    Tweet 457: Bus Fire in Thailand Kills 20 Migrant Workers From Myanmar https://t.co/w3C1SG2hPs
    Tweet 458: The graves used to say "Argentine soldier known only to God." Now they have names. https://t.co/7g98Z0zs12
    Tweet 459: Some species of frogs that vanished may be on the rebound https://t.co/kGYyur9NJm
    Tweet 460: Tech Fix: Want to Purge Your Social Media Timelines? Can You Spare a Few Hours? https://t.co/o0ECSnicLi
    Tweet 461: "How can I get over my sense of betrayal, my rage and my desire to punish this man for the disrespectful way he tre… https://t.co/DdgVwkN3JI
    Tweet 462: "How can I get over my sense of betrayal, my rage and my desire to punish this man for the disrespectful way he tre… https://t.co/PSZnKqivOo
    Tweet 463: Days after researchers announced that a tiny mummy once rumored to be an alien was actually a human infant, Chilean… https://t.co/pWHjcteuAg
    Tweet 464: Tesla once looked like the future of the car industry. But as it burns cash and hits production snags, its own futu… https://t.co/BKoQri2pcC
    Tweet 465: A sprawling exhibition in Amsterdam looks at how a fascination with Japan shaped van Gogh's work https://t.co/ot2sNAyfbR
    Tweet 466: Jews Are Being Murdered in Paris. Again. https://t.co/U8Ios4m8vk
    Tweet 467: Some of the most vocal Parkland seniors — whether accepted or denied to their top college choices — are rethinking… https://t.co/5DjrpLjU0l
    Tweet 468: Some of the most vocal Parkland seniors — whether accepted or denied to their top college choices — are rethinking… https://t.co/ez3RxADRxa
    Tweet 469: President Trump and his old friend David Pecker, whose company owns The National Enquirer, have long had a mutually… https://t.co/SStfyOVJeP
    Tweet 470: RT @nytpolitics: Attorney General Jeff Sessions rejected calls to appoint a second special counsel to investigate the Justice Department su…
    Tweet 471: RT @NYTHealth: There is something ugly inside your rubber ducky, scientists say https://t.co/5JFFHBCv6e
    Tweet 472: A Quick Online Divorce for $60? Not So Fast, Denmark Says https://t.co/mlCHREYMeS
    Tweet 473: RT @koblin: The day after 2016 election, top ABC execs held a meeting to figure out how to program in the Trump era. 18 months later, Rosea…
    Tweet 474: Gym.
    Tan.
    Laundry.
    Check on the kids. 
    https://t.co/gqMNDPIhKL
    Tweet 475: The U.N. secretary general called climate change “the most systemic threat to humankind” https://t.co/6qOyRo5UE4
    Tweet 476: The judge who is presiding at the retrial of Bill Cosby rejected defense motions to recuse himself because his wife… https://t.co/oscBX31Y9D
    Tweet 477: RT @katekelly: Friends of media mogul David Pecker broke bread with Trump at the White House, cementing a crucial link to the Saudi busines…
    Tweet 478: The shake-up in the Veterans Affairs department has brought renewed focus to the debate over privatizing veterans'… https://t.co/eoyrP4aYbc
    Tweet 479: In Opinion,
    
    @rgay writes: "No amount of mental gymnastics can make what Roseanne Barr has said and done in recent… https://t.co/WIAVJ6gjo6
    Tweet 480: Hope Hicks has left the building. Those left behind are wondering what happens now. https://t.co/uadHGL9Ssu
    Tweet 481: RT @mmcintire: Exclusive: How Trump helped tabloid mogul who buried story of alleged affair with Playboy playmate w/@jimrutenberg @katekell…
    Tweet 482: Trump and Pecker have long had a mutually beneficial relationship. 
    
    Pecker's company has championed Trump by buryi… https://t.co/FOQJKEJe2j
    Tweet 483: It is a previously untold chapter in the long, symbiotic relationship between President Trump and tabloid publisher… https://t.co/dCJdxmtZHA
    Tweet 484: The story of one White House dinner shows how the National Enquirer's publisher has used President Trump's friendsh… https://t.co/NpEY3BBXA2
    Tweet 485: RT @jimrutenberg: EXCLUSIVE: Wooing Saudi Business, Tabloid Mogul Had a Powerful Friend: Trump https://t.co/Ovn3jz6PKW
    Tweet 486: The NBA is far more willing to address social issues than other leagues. But the Sacramento Kings' partnership with… https://t.co/udD9uRdeNV
    Tweet 487: Evening Briefing: Here's what you need to know at the end of the day https://t.co/N5NZD3QVgl
    Tweet 488: Some of the most vocal Parkland seniors — whether accepted or denied to their top college choices — are rethinking… https://t.co/W9L5DQiddm
    Tweet 489: Tesla once looked like the future of the car industry. But as it burns cash and hits production snags, its own futu… https://t.co/mV5tZpWRy8
    Tweet 490: The original dream of social media shouldn’t be discarded because of the failures of the current market leaders,… https://t.co/QCRR0pg3zd
    Tweet 491: RT @tiffkhsu: Racist hoodies, offensive skirts, sexist T-shirts, tone-deaf accessories: Retailers repeatedly fail to catch distasteful prod…
    Tweet 492: A martyr for Korean independence, Yu Gwan-sun died at 17 and went on to become a national hero… https://t.co/iNwPlrjCkX
    Tweet 493: RT @nytimesworld: “Our blood is finished, our tears have dried. We will not say another word.” Activists camped at the site of a suicide bo…
    Tweet 494: President Trump criticized Amazon again, causing its stock price to decrease. Here are the facts behind his tweets. https://t.co/7s6qsoPyF2
    Tweet 495: RT @katierogers: During his “infrastructure event,” the president went full on jazz. He said he’d maybe stall the KORUS deal, pull US troop…
    Tweet 496: In Opinion, 
    
    The editorial board writes: "Trump gave no reason for firing Dr. Shulkin, but it’s all too believable… https://t.co/wmdLczlD0U
    Tweet 497: Evening Briefing: Here's what you need to know at the end of the day https://t.co/RB4NEsCcTn
    Tweet 498: RT @inyoungk: For Overlooked I wrote about Yu Gwan-sun, a schoolgirl who became the face of Korea’s 35-year fight for independence against…
    Tweet 499: Derek Jeter’s first inning as chief executive of the Miami Marlins did not quite go as planned https://t.co/SyzubPTPc7
    Tweet 500: The assault on Atlanta’s computers is a vivid example of the perils local governments face in the internet age https://t.co/neclZLUuSy



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
      <td>Fri Mar 30 17:06:00 +0000 2018</td>
      <td>'Change doesn't come from outside.'\n\n#Pilgri...</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBC</td>
      <td>Fri Mar 30 16:26:45 +0000 2018</td>
      <td>❤️ A dying man was granted his final wish in h...</td>
      <td>0.5719</td>
      <td>0.266</td>
      <td>0.734</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBC</td>
      <td>Fri Mar 30 16:02:04 +0000 2018</td>
      <td>The illegal wildlife trade is worth £18bn per ...</td>
      <td>-0.2144</td>
      <td>0.073</td>
      <td>0.810</td>
      <td>0.116</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBC</td>
      <td>Fri Mar 30 15:25:47 +0000 2018</td>
      <td>RT @bbceurovision: 🇬🇧 Presenting our 2018 drea...</td>
      <td>0.8715</td>
      <td>0.377</td>
      <td>0.623</td>
      <td>0.000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBC</td>
      <td>Fri Mar 30 15:10:56 +0000 2018</td>
      <td>RT @BBCNews: Heartbeat actor Bill Maynard dies...</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
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
    BBC        0.065446
    CBS        0.351304
    CNN       -0.120065
    Fox        0.291836
    nytimes   -0.063961
    Name: Compound, dtype: float64




```python
x_axis = np.arange(len(scoresbyorganization))
```


```python
# Build the bar chart for each media source 
plt.figure(figsize = (10,8))

ax = scoresbyorganization.plot(kind='bar')
ax.set_title("Overall Media Sentiment Based on Twitter (%s)" % (time.strftime("%m/%d/%Y")),fontweight='bold')
ax.set_ylabel("Tweet Polarity",fontweight='bold')
ax.set_xticklabels(["BBC", "CBS", "CNN", "Fox", "nytimes"])


rects = ax.patches

for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x()+rect.get_width()/2
    space = 5
    if y_value < 0:
        space *= -3
        va = "top"
    label = "{:.2f}".format(y_value)
    plt.annotate(label,(x_value,y_value),xytext=(0, space),textcoords="offset points",ha="center",va='bottom')


# Save the figure
plt.savefig('Overall Media Sentiment Based on Twitter.png')

# Show plot
plt.show()
```


![png](output_11_0.png)

