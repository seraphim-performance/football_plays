"# football_plays" 
Instructions: 
   Use Stream download browser extension to download games from NFL+.
   Put entire stream of All-22 footage in a directory
   [season]
   -[week]
   --[gameid] -use pbp csv file to find gameid

   Once all videos are in place
   run scripts in this order from [season] directory
   splitscenes.py
   renumvids.py
   pbp2caption.ipynb
   removeprocessed.py
   movescenes.py

   To prep a subset for training run
   prepfactory.py

   You shoul dhave
   [videosdir]
    -[videos]
    -videos.txt
    -labels.txt

   run vivit.py for infernse