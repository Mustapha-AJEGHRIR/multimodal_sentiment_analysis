wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_IAWexEWpH-ly_JaA5EGfZDp-_3flkN1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_IAWexEWpH-ly_JaA5EGfZDp-_3flkN1" -O aesdd.zip && rm -rf /tmp/cookies.txt
unzip aesdd.zip
rm aesdd.zip
mv 'Acted Emotional Speech Dynamic Database' aesdd