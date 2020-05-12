import csv

weebit = csv.writer(open('sample.csv', 'a', newline=''))
newsela = csv.reader(open('sample10Newsela.csv', 'r'))
next(newsela)
for row in newsela:
  weebit.writerow(row)
