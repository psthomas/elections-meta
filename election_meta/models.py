import os
import json
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

## This file pulls out code that is shared between elections-meta.ipynb
## and evaluate.ipynb to avoid code reuse.
## TODO: Consider reorganizing models into classes.
## TODO: Standardize model input and output data structures (and JSON serialization)

# Set plot style
plt.style.use('seaborn-whitegrid')

# DataFrame of categorical ratings for conversions
rating_categories = pd.DataFrame.from_dict({
    'inside_elections': ['currently-safe-democrat', 'democrat-favored', 'lean-democrat', 
        'toss-up-tilt-democrat', 'pure-toss-up', 'toss-up-tilt-republican',
        'lean-republican', 'republican-favored', 'currently-safe-republican'],
    'cnalysis': ['Safe D', 'Likely D', 'Lean D', 'Tilt D', 'Toss-Up', 
        'Tilt R', 'Lean R', 'Likely R', 'Safe R'],
    'dem_margin': [0.34, 0.12, 0.07, 0.04, 0, -0.04, -0.07, -0.12, -0.34]
})

def normed_prob(a, bounds, interval=[0.475, 0.525]): #0.475,0.525 #0.49,0.51
    # Normalize data over [0,1]
    a = (a - bounds[0]) / (bounds[1] - bounds[0])
    # Integrate over interval
    # inbounds = a[(a > interval[0]) & (a < interval[1])]
    # return len(inbounds)/len(a)
    # Use kde instead, avoids threshold problem with discrete data
    kde = stats.gaussian_kde(a)
    return kde.integrate_box_1d(interval[0], interval[1])


def calculate_seat_power(state_metadata):
    '''Creates a hierarchical power sharing model covering all federal and
    state level elected offices. 2020 version, not corrected by year.'''

    total_power = 100
    federal_power = 0.5*total_power
    presidential_power = 0.5*federal_power
    senate_power = 0.25*federal_power
    house_power = 0.25*federal_power

    states_power = 0.5*total_power
    governor_power = 0.5*states_power
    state_senate_power = 0.25*states_power
    state_house_power = 0.25*states_power

    # Calculate state government power values based on fraction of national population
    state_power = state_metadata.copy()
    state_power['multiplier'] = state_power['population']/state_power['population'].sum()
    state_power['governor_power'] = governor_power*state_power['multiplier']
    state_power['state_senate_power'] = state_senate_power*state_power['multiplier']
    state_power['state_house_power'] = state_house_power*state_power['multiplier']
    state_power.drop(['electoral_votes', 'population', 'multiplier'], axis='columns', inplace=True)

    return {'state_power': state_power, 'presidential_power': presidential_power,
        'senate_power': senate_power, 'house_power': house_power,
        'governor_power': governor_power}

def calculate_year_adjusted_seatpower(state_metadata, state_election_frequency):
    '''New for 2022. Adjusts the power for each legislative body/office by the
    length of time it will be held. e.g. Senate is decided for 2 years, presidency
    for 4. State legislative bodies are decided every 2-4 years, varies.'''

    total_power = 100
    federal_power = 0.5*total_power
    presidential_power = 0.5*federal_power*4 # Decided every 4 years
    senate_power = 0.25*federal_power*2 # Body decided every 2
    house_power = 0.25*federal_power*2

    states_power = 0.5*total_power
    cumulative_governor_power = 0.5*states_power*4 # Decided every 4 years
    cumulative_statesenate_power = 0.25*states_power
    cumulative_statehouse_power = 0.25*states_power

    # Calculate state government power values based on fraction of national population,
    # and how often the state legislative bodies hold elections.
    state_power = state_metadata.copy()
    state_power['population_multiplier'] = state_power['population']/state_power['population'].sum()

    governor_power = state_power.copy()
    governor_power.eval(
        'potential_power = @cumulative_governor_power*population_multiplier',
        inplace=True
    )

    statehouse_power = state_power.copy()
    statehouse_power = statehouse_power.merge(
        state_election_frequency.query('branch == "statehouse"'), 
        on='state'
    )
    statehouse_power.eval(
        'potential_power = @cumulative_statehouse_power*population_multiplier*election_frequency',
        inplace=True
    )
    
    statesenate_power = state_power.copy()
    statesenate_power = statesenate_power.merge(
        state_election_frequency.query('branch == "statesenate"'), 
        on='state'
    )
    statesenate_power.eval(
        'potential_power = @cumulative_statesenate_power*population_multiplier*election_frequency', 
        inplace=True
    )

    return {'statehouse_power': statehouse_power, 'statesenate_power': statehouse_power,
        'governor_power': governor_power, 'presidential_power': presidential_power,
        'ussenate_power': senate_power, 'ushouse_power': house_power}


## Presidential, processing economist output

def presidential_tipping_point(ec, state_metadata):
    '''Find tipping point state for each simulation, return dataframe
    including tipping probability.'''
    
    votes = ec.loc[:,'AK':'WY'].T
    ev = state_metadata[['state_abbr', 'electoral_votes_2016']].copy()
    ev.rename(columns={'electoral_votes_2016': 'ev'}, inplace=True)
    ev.set_index('state_abbr', inplace=True)
    tip_states = []

    for col in votes.columns:
        a = votes[col]
        onesim = pd.concat([ev, a], axis=1)
        onesim.rename(columns={col:'dem'}, inplace=True)
        onesim['win'] = (onesim['dem'] > 0.5)
        dem_ev = (onesim['ev']*onesim['win']).sum()
        if abs(dem_ev - 269) < 0.025*538: #20, 50 +/-0.025
        #if abs(269-dem_ev) < 100: #55
        #if dem_ev >= 269: #greater equal, or greater? 
            if dem_ev + np.random.choice([0,1]) > 269: #Half time give tie win to D/R
                onesim.sort_values(by='dem', ascending=False, inplace=True)
            else:
                onesim.sort_values(by='dem', ascending=True, inplace=True)
            onesim['evsum'] = onesim['ev'].cumsum()
            tip_state = onesim[onesim['evsum'] >= 269].head(1).index[0]
            tip_states.append(tip_state)
    
    tp = pd.DataFrame.from_dict({'state':tip_states, 'count': 1})
    tp = tp.groupby(by='state').agg({'count':'sum'})
    tp['pr_tip'] = tp['count']/tp['count'].sum()
    del tp['count']
    tp.reset_index(drop=False, inplace=True)
    return tp

def get_presidential_power(tp, presidential_power, ec_prob_close, state_metadata):
    '''Calculate presidential power by state using baseline power value,
    the probability of a close election, and the tipping point probability'''
    
    tp = tp.copy() # Create local copy
    tp['realized_power'] = presidential_power*ec_prob_close*tp['pr_tip']
    missing_states = state_metadata[~state_metadata['state_abbr'].isin(tp.state)]['state_abbr']
    missing = pd.DataFrame.from_dict({
        'state': missing_states,
        'pr_tip': 0,
        'realized_power': 0
    })
    cols = ['state', 'office', 'district', 'potential_power', 'pr_close', 'pr_tip', 'realized_power']
    pres_power_df = pd.concat([tp, missing])
    pres_power_df[['office', 'district', 'potential_power', 'pr_close']] = \
        ['president', None, presidential_power, ec_prob_close]
    pres_power_df = pres_power_df[cols]
    pres_power_df.sort_values(by='realized_power', ascending=False, inplace=True)
    return pres_power_df


## Presidential, my own model for evaluation because I don't 
## have all the economist source code.
# Mostly borrowed from Drew Linzer's model that I modified here:
# https://github.com/psthomas/election-sim

# state_data is a df with the predicted dem two party margin,
# and the state as an index.
#             dem
# state          
# AK     0.482267
# AL     0.382979
# AZ     0.521030
# CA     0.693744
# CO     0.576441
def simulate_president(n, state_uncertainty_sd, national_uncertainty_sd, state_data, electoral_votes):

    dem_state_wins = pd.Series(0, index = state_data.index)
    dem_state_votes = pd.DataFrame()
    dem_ev_sim = []
    tipping_point_states = []

    for sim in np.arange(n):
        # simulate 51 state-level election outcomes
        national_error = np.random.normal(0, national_uncertainty_sd)
        # Try using a student's T distribution here?
        # national_error = stats.t.rvs(df=4, loc=0, scale=national_uncertainty_sd)
        # 'dem_vote': stats.t.rvs(df=4, loc=state_data.dem, scale=state_uncertainty_sd) + national_error
        one_simulation = pd.DataFrame({
            'state': state_data.index,
            'dem_vote': np.random.normal(state_data.dem, state_uncertainty_sd) + national_error
        }).set_index('state')

        # did the democrat win each state in this simulation?
        one_simulation['dem_win'] = (one_simulation.dem_vote > 0.5)+0

        # record simulation results
        one_simulation = one_simulation.merge(electoral_votes, left_index = True, right_index = True)
        dem_state_wins = dem_state_wins + one_simulation.dem_win
        dem_state_votes = dem_state_votes.append(one_simulation.dem_vote)
        one_simulation['dem_ev'] = one_simulation.dem_win * one_simulation.EV
        dem_ev = one_simulation['dem_ev'].sum()
        dem_ev_sim.append(dem_ev)
        # 20 point threshold for tipping? How is this done?
        # This assumption has huge implications for the model. pr_tip|close, or just pr_tip?
        # Define "close" as 5% around central outcome = 0.05*538 = 26.9 electoral votes
        # then you're multiplying pr_close*pr_tip|close, which might make more sense.
        # Problem is that your use of the economist model calculated pr_tip using all elections.
        # Or just find tipping point for all simulations. I found this works best comparatively
        # when processing the economists simulations, but not for my own model.
        # If you do change this, you need to update the Senate and House models to act accordingly
        # The Govs/State Legislatures do not depend on tipping point probabilities.
        if abs(dem_ev - 269) < 0.025*538: #20, 50  +/-2.5% from center
            # Find tipping point state, sorting after vector addition above
            if dem_ev > 269:
                one_simulation = one_simulation.sort_values(by='dem_vote', ascending=False)
            else:
                # Calculate the R Winner tipping point
                one_simulation = one_simulation.sort_values(by='dem_vote', ascending=True)
            one_simulation['cumulative_votes'] = one_simulation.EV.cumsum()
            tipping_point_state = one_simulation[one_simulation['cumulative_votes'] >= 270].head(1).index[0]
            tipping_point_states.append(tipping_point_state)

    # Calculate summary statistics. Mean, quantiles, and tipping probabilities by state
    summary = pd.concat([
        dem_state_votes.quantile([0.025, 0.975]),
        dem_state_votes.agg(['mean']) 
    ])
    summary = summary.T
    summary = summary.reset_index(drop=False)
    summary = summary.rename(columns={'index':'state', 0.025:'q_025', 0.975: 'q_975'})

    # Tipping probabilities by state
    tp = pd.DataFrame.from_dict({'state':tipping_point_states, 'count': 1})
    tp = tp.groupby(by='state').agg({'count':'sum'})
    tp['pr_tip'] = tp['count']/tp['count'].sum()
    del tp['count']
    tp.reset_index(drop=False, inplace=True)

    summary = summary.merge(tp, on='state', how='left')
    summary['pr_tip'] = summary['pr_tip'].fillna(0) # States with no chance of tipping
    summary['district'] = None
    summary = summary[['state', 'mean', 'q_025', 'q_975', 'pr_tip']]

    return {'dem_state_wins' : dem_state_wins, 'dem_state_votes': dem_state_votes,
       'dem_ev_sim': dem_ev_sim, 'tipping_point_states': tipping_point_states, 
       'summary': summary}


## Senate, my own evaluation model as well as I don't know enough
## to run McCartan's
def simulate_senate(n, state_uncertainty_sd, national_uncertainty_sd, senate_data, safe_dem_seats): 
    seats = senate_data[['state', 'district']]
    sims = []
    dem_seats = []
    tip_seats = []
    summary = []

    for i in range(n):
        # Account for potential national swing
        national_error = np.random.normal(0, national_uncertainty_sd)
        
        onesim = senate_data[['state', 'district', 'dem']].copy()
        onesim['sample'] = np.random.normal(onesim['dem'], state_uncertainty_sd) + national_error
        onesim['sample'] = onesim['sample'].clip(lower=0.0, upper=1.0)
        dem_sim_seats = (onesim['sample'] > 0.5).sum() + safe_dem_seats #Add uncontested seats
        sims.append(onesim['sample'])
        dem_seats.append(dem_sim_seats)
        # Find tipping point seat
        if abs(dem_sim_seats - 50) < 0.025*100: # Limit to close races
            if dem_sim_seats >= 50: # Treats a tie as a dem win
                onesim = onesim.sort_values(by='sample', ascending=False)
                tip_num = 50 - safe_dem_seats # 50 to majority
            else:
                onesim = onesim.sort_values(by='sample', ascending=True)
                safe_rep_seats = 100 - safe_dem_seats - len(onesim['dem'])
                tip_num = 50 - safe_rep_seats # Should this be 51 to majority w/o president?
            onesim.reset_index(drop=True, inplace=True)
            # The threshold seat that must be won by either party for a majority
            tip_seats.append(list(onesim.loc[tip_num][['state', 'district']]))

    
    # Process simulation results to get seat statistics
    sims = np.array(sims)
    seat_sims = sims.T
    for i, seat in enumerate(seats.itertuples()):
        this_sim = seat_sims[i]
        mean_vote = np.mean(this_sim)
        quantile = np.quantile(this_sim, [0.025, 0.975])
        summary.append({'state': getattr(seat,'state'), 'district': getattr(seat, 'district'), 
            'mean_vote': mean_vote, 'q_025': quantile[0], 'q_975': quantile[1]})

    # Tipping probabilities by senate seat
    tp = pd.DataFrame.from_dict({
        'state': [i[0] for i in tip_seats],
        'district': [i[1] for i in tip_seats],
        'count': 1})
    tp = tp.groupby(by=['state', 'district']).agg({'count':'sum'})
    tp['pr_tip'] = tp['count']/tp['count'].sum()
    del tp['count']
    tp.reset_index(drop=False, inplace=True)

    # Make summary into dataframe, merge with tips, return dataframe instead.
    # If you eventually output the model, mabye convert df to json then?
    summary = pd.DataFrame.from_dict(summary)
    summary = summary.merge(tp, on=['state', 'district'], how='left') #GA has special election GA-S
    summary['pr_tip'] = summary['pr_tip'].fillna(0) # Seats with no chance of tipping
    summary = summary[['state', 'district', 'mean_vote', 'q_025', 'q_975', 'pr_tip']]
            
    return {'seats': seats, 'sims': sims, 'dem_seats': dem_seats,
        'tip_seats': tip_seats, 'summary': summary}


## House, my own simulations

def simulate_house(n, state_uncertainty_sd, national_uncertainty_sd, house_data): 
    seats = house_data[['state', 'district']]
    sims = []
    dem_seats = []
    tip_seats = []
    summary = []

    for i in range(n):
        # Account for potential national swing
        national_error = np.random.normal(0, national_uncertainty_sd)
        
        onesim = house_data[['state', 'district', 'dem']].copy()
        onesim['sample'] = np.random.normal(onesim['dem'], state_uncertainty_sd) + national_error
        onesim['sample'] = onesim['sample'].clip(lower=0.0, upper=1.0)
        dem_sim_seats = (onesim['sample'] > 0.5).sum()
        sims.append(onesim['sample'])
        dem_seats.append(dem_sim_seats)
        # Find tipping point seat
        if abs(dem_sim_seats - 217.5) < 0.05*435: # Limit to close races? < 40
            if dem_sim_seats >= 218:
                onesim = onesim.sort_values(by='sample', ascending=False)
            else:
                onesim = onesim.sort_values(by='sample', ascending=True)
            onesim.reset_index(drop=True, inplace=True)
            tip_seats.append(list(onesim.loc[218][['state', 'district']]))
    
    # Process simulation results to get seat statistics
    sims = np.array(sims)
    seat_sims = sims.T
    for i, seat in enumerate(seats.itertuples()):
        this_sim = seat_sims[i]
        mean_vote = np.mean(this_sim)
        quantile = np.quantile(this_sim, [0.025, 0.975])
        summary.append({'state': getattr(seat,'state'), 'district': getattr(seat, 'district'), 
            'mean_vote': mean_vote, 'q_025': quantile[0], 'q_975': quantile[1]})
            
    return {'seats': seats, 'sims': sims, 'dem_seats': dem_seats,
        'tip_seats': tip_seats, 'summary': summary}

def get_house_power(house_results, house_power, house_prob_close, house_data):
    #TODO: Consider moving this to summary code in model
    house_tip = pd.DataFrame(
        columns=['state', 'district'], 
        data=house_results['tip_seats']
    )
    house_tip['count'] = 1
    house_tip = house_tip.groupby(by=['state', 'district']).agg({'count': 'sum'})
    house_tip['pr_tip'] = house_tip['count']/house_tip['count'].sum()
    house_tip['realized_power'] = house_power*house_prob_close*house_tip['pr_tip']
    house_tip.reset_index(drop=False, inplace=True)
    del house_tip['count']
    #house_tip.sort_values(by='realized_power', ascending=False).head(30)

    missing = house_data[['state', 'district']]
    house_power_df = missing.merge(house_tip, on=['state', 'district'], how='outer')
    house_power_df.fillna(value=0, inplace=True)
    cols = ['state', 'office', 'district', 'potential_power', 'pr_close', 'pr_tip', 'realized_power']
    house_power_df[['office', 'potential_power', 'pr_close']] = \
        ['ushouse', house_power, house_prob_close]
    house_power_df = house_power_df[cols]
    house_power_df.sort_values(by='realized_power', ascending=False, inplace=True)
    return house_power_df


## Governor model

def simulate_governors(n, state_uncertainty_sd, national_uncertainty_sd, govs): 
    seats = govs['state']
    votes = []

    for i in range(n):

        if national_uncertainty_sd:
            # Account for potential national swing
            national_error = np.random.normal(0, national_uncertainty_sd)
        else:
            national_error = 0
        
        # Only need to sample from seat, adding in national swing
        # state_uncertainty_sd can either be a constant, or an array in numpy call,
        # so pass an array using 1/2 the governors sd or se?
        # Maybe include an option to exclude the national swing if e.g. using 538 bounds
        # which should include that already.
        sample = np.random.normal(govs['dem'], state_uncertainty_sd) + national_error
        sample = np.clip(sample, 0.0, 1.0)
        votes.append(sample)
    
    # Process simulation results to get seat statistics
    state_votes = np.array(votes).T
    summary = []
    for n, seat in enumerate(seats):
        seat_votes = state_votes[n]
        mean_vote = np.mean(seat_votes)
        pr_close = normed_prob(seat_votes, bounds=[0,1])
        quantile = np.quantile(seat_votes, [0.025, 0.975])
        summary.append({'state': seat, 'pr_close': pr_close, 'mean_vote': mean_vote,
            'q_025': quantile[0], 'q_975': quantile[1]})

    return {'seats': seats, 'votes': votes, 'summary': summary}

def get_governor_power(gov_power_df, state_power):
    cols = ['state', 'office', 'district', 'potential_power', 'pr_close', 'pr_tip', 'realized_power']
    gov_seat_power = state_power[['state_abbr', 'governor_power']].copy()
    gov_seat_power.rename(
        columns={'state_abbr': 'state', 'governor_power': 'potential_power'}, 
        inplace=True
    )
    gov_power_df = gov_power_df.merge(gov_seat_power, on='state')
    gov_power_df['realized_power'] = gov_power_df['pr_close']*gov_power_df['potential_power']
    gov_power_df[['pr_tip', 'office', 'district']] = [1, 'governor', None]
    gov_power_df = gov_power_df[cols]
    gov_power_df.sort_values(by='realized_power', ascending=False, inplace=True)
    return gov_power_df


## State House Model

def prep_statehouse_data(path, rating_categories, stateleg_metadata):
    sh = pd.read_csv(path)
    sh.columns = map(str.lower, sh.columns)
    sh = sh[sh['rating'] != 'No Election']
    sh = sh[['state', 'rating']]

    sh = sh.merge(rating_categories[['cnalysis', 'dem_margin']],
        right_on = 'cnalysis', left_on = 'rating'
    )
    sh = sh.merge(stateleg_metadata[['state_abbr', 'house_num']],
        left_on = 'state',right_on = 'state_abbr'
    )
    sh['dem'] = sh['house_num']/2 + sh['house_num']*(sh['dem_margin']/2)
    return sh

def simulate_statehouses(n, state_uncertainty_sd, national_uncertainty_sd, house_data):
    '''Note: this simulates expected seat count,
    not expected democratic margin by state'''
    
    house_data = house_data.copy()
    seats = house_data['state']
    sims = []
    summary = []
    # Scale the uncertainty so it is in units of house seats
    house_data['scaled_sd'] = house_data['house_num']*state_uncertainty_sd
    # 2022: Note, this isn't a great approach because e.g. you could have large
    # polling uncertainty, zero competitive seats, and very low seat share uncertainty.
    # Better approach is to just use uncertainty from actual longitudinal seat share 
    # data. Not perfect, but better than this.

    for i in range(n):
        onesim = house_data[['state', 'house_num', 'dem', 'scaled_sd']].copy()
        # Account for potential national swing, but scale that swing to 
        # the number of seats in each state house
        national_error = np.random.normal(0, national_uncertainty_sd)*onesim['house_num']
        onesim['sample'] = np.random.normal(onesim['dem'], onesim['scaled_sd']) + national_error
        onesim['sample'] = np.round(onesim['sample']) # Make discrete? Valid? Doesn't seem valid,
        # the states with small number of discrete seats like AZ get a lot more probability cut
        # off from the pr_close interval this way. Not sure the solution, I should probably ask
        # on stats stackexchange. Current solution is to use KDE to integrate.
        # Should not be clipped to 1 because you're modeling number of seats, not voteshare
        # onesim['sample'] = onesim['sample'].clip(lower=0.0, upper=1.0)
        sims.append(onesim['sample'])
    
    # Process simulation results to get seat statistics
    sims = np.array(sims)
    seat_sims = sims.T
    house_data.set_index('state', inplace=True)
    for i, seat in enumerate(seats):
        res = seat_sims[i]
        pr_close = normed_prob(res,
            bounds=[0, house_data.loc[seat, 'house_num']]
        ) #interval=[0.475, 0.525]
        mean_seats = np.mean(res)
        quantile = np.quantile(res, [0.025, 0.975])
        summary.append({'state': seat, 'pr_close': pr_close,
            'mean_seats': mean_seats, 'q_025': quantile[0], 'q_975': quantile[1]})
    
    return {'seats': seats, 'sims': sims, 'summary': summary}

def get_statehouse_power(statehouse_summary, state_power):
    sthspw = statehouse_summary[['state', 'pr_close']]
    sthspw = sthspw.merge(
        state_power[['state_abbr','state_house_power']].copy() \
            .rename(columns={'state_abbr':'state', 'state_house_power':'potential_power'})
    )
    sthspw['realized_power'] = sthspw['potential_power']*sthspw['pr_close']
    #sthspw[['office', 'district', 'offi']]
    sthspw.sort_values(by='realized_power', ascending=False, inplace=True)
    sthspw[['office', 'district', 'pr_tip']] = ['statehouse', None, 1]
    cols = ['state', 'office', 'district', 'potential_power', 'pr_close', 'pr_tip', 'realized_power']
    sthspw = sthspw[cols]
    return sthspw


## State Senates

def prep_statesenate_data(path, rating_categories, stateleg_metadata):
    ss = pd.read_csv(path)
    ss.columns = map(str.lower, ss.columns)
    ss = ss[ss['rating'] != 'No Election']
    ss = ss[['state', 'rating']]

    ss = ss.merge(rating_categories[['cnalysis', 'dem_margin']],
        right_on = 'cnalysis', left_on = 'rating'
    )
    ss = ss.merge(stateleg_metadata[['state_abbr', 'senate_num']],
        left_on = 'state',right_on = 'state_abbr'
    )
    ss['dem'] = ss['senate_num']/2 + ss['senate_num']*(ss['dem_margin']/2)
    ss = ss.sort_values(by='dem_margin')
    return ss

def simulate_statesenates(n, state_uncertainty_sd, national_uncertainty_sd, senate_data):
    '''Note: this simulates expected seat count,
    not expected democratic margin by state.'''
    
    senate_data = senate_data.copy()
    states = senate_data['state']
    sims = []
    summary = []
    # Scale the uncertainty so it is in units of house seats
    senate_data['scaled_sd'] = senate_data['senate_num']*state_uncertainty_sd

    for i in range(n):
        onesim = senate_data[['state', 'senate_num', 'dem', 'scaled_sd']].copy()
        # Account for potential national swing, but scale that swing to 
        # the number of seats in each state senate
        national_error = np.random.normal(0, national_uncertainty_sd)*onesim['senate_num']
        onesim['sample'] = np.random.normal(onesim['dem'], onesim['scaled_sd']) + national_error
        onesim['sample'] = np.round(onesim['sample']) # Make discrete? Valid?
        #onesim['sample'] = onesim['sample'].clip(lower=0.0, upper=1.0)
        sims.append(onesim['sample'])
    
    # Process simulation results to get seat statistics
    sims = np.array(sims)
    seat_sims = sims.T
    senate_data.set_index('state', inplace=True)
    for i, state in enumerate(states):
        res = seat_sims[i]
        pr_close = normed_prob(
            res,
            bounds=[0, senate_data.loc[state, 'senate_num']]
        ) #interval=[0.475, 0.525]
        mean_seats = np.mean(res)
        quantile = np.quantile(res, [0.025, 0.975])
        summary.append({'state': state, 'pr_close': pr_close,
            'mean_seats': mean_seats, 'q_025': quantile[0], 'q_975': quantile[1]})
    
    return {'seats': states, 'sims': sims, 'summary': summary}

def get_statesenate_power(summary, state_power):
    stspw = summary[['state', 'pr_close']].copy()
    stspw = stspw.merge(
        state_power[['state_abbr','state_senate_power']].copy() \
            .rename(columns={'state_abbr':'state', 'state_senate_power':'potential_power'})
    )
    stspw['realized_power'] = stspw['potential_power']*stspw['pr_close']
    #sthspw[['office', 'district', 'offi']]
    stspw.sort_values(by='realized_power', ascending=False, inplace=True)
    stspw[['office', 'district', 'pr_tip']] = ['statesenate', None, 1]
    cols = ['state', 'office', 'district', 'potential_power', 'pr_close', 'pr_tip', 'realized_power']
    stspw = stspw[cols]
    return stspw


## Visualizations

def plot_histogram(sims, mean_outcome, midpoint, title, x_label, outcome=None, bins=None, out=None):
    fig, ax = plt.subplots(figsize=[6, 4])
    if bins:
        #kde=False, norm_hist=False
        # distplot is deprecated
        # sns.distplot(sims, ax=ax, bins=bins)
        sns.histplot(sims, ax=ax, bins=bins, stat='density', kde=True, element='step', edgecolor=None)
    else:
        #sns.distplot(sims, ax=ax)
        sns.histplot(sims, ax=ax, stat='density', kde=True, element='step', edgecolor=None)
    #ax.grid(alpha=0.5)
    ax.axvline(mean_outcome, color='red', alpha=0.9)
    ax.axvline(midpoint, color='gray')
    if outcome:
        ax.axvline(outcome, color='darkgreen')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if out:
        plt.savefig(out, bbox_inches='tight') #bbox_inches='tight'
    return ax

def plot_seatprob(x, y, mean_outcome, midpoint, title, x_label, y_label, outcome=None, bins=None, out=None):
    fig, ax = plt.subplots(figsize=[6, 4])

    sns.lineplot(x=x, y=y, ci=None, ax=ax)
    # Discrete bars are accurate, but jumbled
    # sns.barplot(x=x, y=y, ci=None, ax=ax)
    ax.axvline(mean_outcome, color='red', alpha=0.9)
    ax.axvline(midpoint, color='gray')
    if outcome:
        ax.axvline(outcome, color='darkgreen')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.fill_between(x, y, alpha=0.2)
    ax.set(ylim=(0, y.max()*1.05))
    if out:
        plt.savefig(out, bbox_inches='tight') #bbox_inches='tight'
    return ax


def seatplot(plot_df, x, y, title, x_label, midpoint=0.5, figsize=[6, 11], xlim=[0.2, 0.8], outcome=None, out=None):
    # https://matplotlib.org/3.1.1/gallery/statistics/errorbar_features.html
    # https://stackoverflow.com/questions/31081568
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        x=plot_df[x],
        y=plot_df.index,
        xerr=np.array(plot_df[['lower', 'upper']]).T,
        fmt='.' #b
    ) # alpha=0.7 color='cornflowerblue' color='royalblue'  alpha=0.8
    # Join the outcome in advance as an 'outcome' column in plot_df
    if outcome:
        ax.scatter(
            x=plot_df['dem_result'],
            y=plot_df.index,
            marker='.',
            c='darkgreen'
        )
    plt.yticks(ticks=plot_df.index, labels=plot_df[y])
    ax.grid(alpha=0.6)
    ax.axvline(midpoint, color='gray', alpha=0.9)
    ax.set_xlabel(x_label)
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    plt.title(title)
    if out:
        plt.savefig(out, bbox_inches='tight') #bbox_inches='tight' , quality=95
    return ax

def output_table(src, outpath, center=True):
    '''Outputs a table to be embedded as iframe with my blog styling.
    Works with preformatted styler objects or dataframes.
    '''
    # https://stackoverflow.com/questions/36897366
    # https://stackoverflow.com/a/47723330
    # https://stackoverflow.com/a/31513163
    # Setting table styles:
    # https://www.geeksforgeeks.org/display-the-pandas-dataframe-in-table-style-and-border-around-the-table-and-not-around-the-rows/
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.set_table_styles.html

    template = '''
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>

    table {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 12px;
    }
    table, th, td {
        border-top: 1px solid #ddd;
        border-bottom: 1px solid #ddd;
        border-collapse: collapse;
    }
    th, td {
        padding: 8px;
        text-align: left;
        font-family: Helvetica, Arial, sans-serif;
    }
    th {
        background-color: #f2f2f2;
    }
    </style>
    </head>
    <body>
    ||0||
    </body>
    </html>
    '''
    # text-align: center;
    # margin-left: auto;
    # margin-right: auto;
    # <link rel="stylesheet" href="https://pstblog.com/assets/css/main.css">
    if type(src) == pd.core.frame.DataFrame:
        src = src.style
    if center:
        # TODO: this modifies the styler object in place. OK?
        src.set_table_styles(
            [{'selector' : '', 'props' : [('margin', '0px auto')]}]
        )
    result = template.replace("||0||", src.render())
    
    with open(outpath, 'w') as f:
        f.write(result)

def output_interactive_table(table, datatable_def, outpath):
    '''Outputs an interactive dataTable to be embedded as iframe.
    Works with rendered html tables and a dataTable definition.
    '''
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <link type="text/css" rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
    <style>
    body {font-family:sans-serif;font-size:12px;}
    </style
    </head>
    <body>

    <div>
    ||table||
    </div>

    <script>
     $(document).ready( function () {
         $('table[id^=T_]').DataTable(||datatable_def||);
     });
    </script>

    </body>
    </html>
    '''

    table_html = html_template.replace("||table||", table)
    table_html = table_html.replace("||datatable_def||", datatable_def)
    
    with open(outpath, 'w') as f:
        f.write(table_html)

