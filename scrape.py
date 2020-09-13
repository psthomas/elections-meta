import datetime
import json
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import xml.etree.ElementTree as et
import re

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

####### Current Scrapers ##########

def get_economist(today):
    economist_url = 'https://cdn.economistdatateam.com/us-2020-forecast/data/president/economist_model_output.zip'
    # Weird, but has empty directory in path
    ec_filepath = 'output/site_data//electoral_college_simulations.csv'
    economist_archive = Path('data/archive/economist_electoral_college_simulations_{0}.csv'.format(today.isoformat()))
    economist_out = Path('data/president/economist_electoral_college_simulations.csv')

    toplines_filepath = 'output/site_data//state_averages_and_predictions_topline.csv'
    economist_topline_archive = Path('data/archive/economist_state_averages_and_predictions_topline_{0}.csv'.format(today.isoformat()))
    economist_topline_out = Path('data/president/economist_state_averages_and_predictions_topline.csv')

    # https://stackoverflow.com/questions/5710867
    resp = requests.get(economist_url)
    zipfile = ZipFile(BytesIO(resp.content))

    # Pull just the simulation csv out of zip
    ec_file = zipfile.read(ec_filepath) 
    economist_archive.write_bytes(ec_file)
    economist_out.write_bytes(ec_file)

    # Pull toplines out of zip
    toplines_file = zipfile.read(toplines_filepath)
    economist_topline_archive.write_bytes(toplines_file)
    economist_topline_out.write_bytes(toplines_file)
    print('Accessed President:', economist_url)

def get_mccartan(today):
    senate_url = 'https://raw.githubusercontent.com/CoryMcCartan/senate/master/docs/estimate.json'
    senate = json.loads(requests.get(senate_url).content)
    archive = 'data/archive/mccartan_senate_{0}.json'.format(today)
    outpath = 'data/senate/mccartan_senate.json'
    with open(archive, 'w') as a, open(outpath, 'w') as f:
        json.dump(senate, a)
        json.dump(senate, f)

    senate_sim_url = 'https://raw.githubusercontent.com/CoryMcCartan/senate/master/docs/sims.json'
    senate_sim = json.loads(requests.get(senate_sim_url).content)
    sim_archive = 'data/archive/mccartan_senate_sim_{0}.json'.format(today)
    sim_outpath = 'data/senate/mccartan_senate_sim.json'
    with open(sim_archive, 'w') as a, open(sim_outpath, 'w') as f:
        json.dump(senate_sim, a)
        json.dump(senate_sim, f)

    print('Accessed Senate:', senate_url)

def get_inside_elections_house(today):
    house_url = 'http://www.insideelections.com/api/xml/ratings/house'
    xml = requests.get(house_url)
    root = et.fromstring(xml.content)
    races = root.findall('race')
    house_data = {'state':[], 'district':[], 'party':[], 'rating':[]}
    rows = []
    for race in races:
        house_data['state'].append(race.find('state').text)
        house_data['party'].append(race.find('party').text)
        house_data['district'].append(race.find('district').text)
        house_data['rating'].append(race.find('rating').find('segment').text)

    house_data = pd.DataFrame.from_dict(house_data)
    archive = 'data/archive/inside_elections_house_{0}.csv'.format(today)
    outpath = 'data/house/inside_elections_house.csv'
    house_data.to_csv(archive, index=False)
    house_data.to_csv(outpath, index=False)
    print('Accessed House:', house_url)


def get_inside_elections_govs(today):
    gov_url = 'http://www.insideelections.com/api/xml/ratings/governor'
    xml = requests.get(gov_url)
    root = et.fromstring(xml.content)
    races = root.findall('race')
    gov_data = {'state':[], 'party':[], 'rating':[]}
    rows = []
    for race in races:
        gov_data['state'].append(race.find('state').text)
        gov_data['party'].append(race.find('party').text)
        gov_data['rating'].append(race.find('rating').find('segment').text)

    gov_data = pd.DataFrame.from_dict(gov_data)
    archive = 'data/archive/inside_elections_gov_{0}.csv'.format(today)
    outpath = 'data/governor/inside_elections_gov.csv'
    gov_data.to_csv(archive, index=False)
    gov_data.to_csv(outpath, index=False)
    print('Accessed Governors:', gov_url)
        

def get_cnalysis(today):
    sheet_url = 'https://docs.google.com/spreadsheets/d/1ECH6skQbXMZMZ8ZZKcL5rvwfhg_h2GCR1yjGxLU2q_Y/export?format=csv&gid={0}'
    statehouse_url = sheet_url.format('1171481198')
    statesenate_url = sheet_url.format('611135353')

    statehouse_archive = Path('data/archive/cnalysis_shodds_{0}.csv'.format(today))
    statehouse_out = Path('data/statehouse/cnalysis_shodds.csv')
    statesenate_archive = Path('data/archive/cnalysis_ssodds_{0}.csv'.format(today))
    statesenate_out = Path('data/statesenate/cnalysis_ssodds.csv')

    house_res = requests.get(statehouse_url).content
    senate_res = requests.get(statesenate_url).content

    statehouse_archive.write_bytes(house_res)
    statehouse_out.write_bytes(house_res)
    statesenate_archive.write_bytes(senate_res)
    statesenate_out.write_bytes(senate_res)

    print('Accessed State Houses:', statehouse_url)
    print('Accessed State Senates:', statesenate_url)


###### Unused scrapers #########

def get_politico(today):
    urls = {
        'senate': 'https://www.politico.com/2020-election/race-forecasts-and-predictions/senate/', 
        'house': 'https://www.politico.com/2020-election/race-forecasts-and-predictions/house/',
        'governor': 'https://www.politico.com/2020-election/race-forecasts-and-predictions/governor/'
    }

    for seat in urls.keys():
        html = requests.get(urls[seat]).content
        soup = BeautifulSoup(html, 'html.parser')
        # For some reason id isn't identified by bs4 for this script
        # data = soup.find('script',  id = ' __NEXT_DATA__')
        data = soup.find('script', {'type': 'application/json'})
        data = json.loads(data.string)
        df_data = data['props']['pageProps']['rawRatings']['ratings']

        ratings = pd.DataFrame(
            data['props']['pageProps']['rawRatings']['ratings'],
            columns=['rating_category', 'label', 'id', 'body', 'state']
        )

        # Archive full json record
        seat_archive = 'data/archive/politico_{0}_{1}.json'.format(seat, today)
        with open(seat_archive, 'w') as f:
            json.dump(data, f)

        # Write out parsed csv
        ratings.to_csv('data/{0}/politico_{0}.csv'.format(seat), index=False)

        print('Accessed {0}: {1}'.format(seat, urls[seat]))

def process_politico(data):
    ratings = pd.DataFrame(
        data['props']['pageProps']['rawRatings']['ratings'],
        columns=['rating_category', 'label', 'id', 'body', 'state', 'is_special']
    )

    return ratings

def load_politico(day):
    for seat in ['senate', 'house', 'governor']:
        archive = 'data/archive/politico_{0}_{1}.json'.format(seat, day)
        outpath = 'data/{0}/politico_{0}.csv'.format(seat)
        with open(archive, 'r') as f:
            data = json.loads(f.read())

        ratings = process_politico(data)
        print(ratings.head())
        ratings.to_csv(outpath, index=False)


def scrape_ballotpedia():
    states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
        'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 
        'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 
        'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
        'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 
        'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 
        'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    states = [x.replace(' ', '_') for x in states]
    states_regexs = [re.compile('{0}_(?!Legislature)(?!Constitution)'.format(x)) for x in states]

    #result = requests.get('https://ballotpedia.org/State_legislature')
    #soup = BeautifulSoup(result.content, 'html.parser')

    with open('data/Statelegislature-Ballotpedia.html', 'r') as f:
        text = f.read()

    soup = BeautifulSoup(text, 'html.parser')

    state_links = []

    link_re = re.compile('^/.*(Senate|House|State_Assembly)')

    for link in soup.findAll('a', attrs={'href': link_re}):  #re.compile("^/")
        state_links.append(link.get('href'))
        # for state_regex in states_regexs:
        #     href = link.get('href')
        #     # print(href, regex)
        #     # return
        #     if state_regex.search(href):
        #         state_links.append(href)

    return list(set(state_links))  #Filter duplicates
    # /html/body/div[7]/div[2]/div/div/div[2]/div[4]/table[3]/tbody/tr[1]/td[1]


if __name__ == '__main__':
    today = datetime.date.today()

    # Access economist presidential model output
    get_economist(today)

    # Get McCartan Senate model
    get_mccartan(today)

    # Inside Elections governor and House seat ratings
    get_inside_elections_govs(today)
    get_inside_elections_house(today)

    # CNalysis state legislature ratings
    get_cnalysis(today)
