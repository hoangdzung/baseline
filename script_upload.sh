
if [ -d $1 ]; then
    IFS='/'
    read -a strarr <<< $1
    i=`expr ${#strarr[*]} - 1`
    aws s3 cp $1 s3://graphframes-sh2/dataout/"${strarr[${i}]}" --recursive
else
    aws s3 cp $1 s3://graphframes-sh2/dataout/
fi
