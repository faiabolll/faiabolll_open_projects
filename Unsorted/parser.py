from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from time import sleep
import datetime
import re
import json
import os
import random
from constants import *
# for debugging
import pprint
# sql
import psycopg2 as sql_db
from googletrans import Translator
from pymongo import MongoClient
# telegram alert
import telebot

try:
    telebot.apihelper.proxy = {'https': 'https://' + PROXY}
    bot = telebot.TeleBot(T_API)
except:
    TG_CONNECTION = False
else:
    TG_CONNECTION = True

translator = Translator()


class ProxyList():
    def __init__(self, path):
        with open(path, 'r') as f:
            proxies = f.readlines()
            proxies = [re.sub(r'\t', ':', p) for p in proxies]
            proxies = [re.sub(r'\n', '', p) for p in proxies]
            self.proxy_list = proxies

    def get_random_proxy(self):
        """
		Return randomly chosen proxy from proxy list
		"""
        return random.choice(self.proxy_list)



class Database():
    def __init__(self, mode="json"):
        self.db_type = mode
        self.is_connected = False

    def store(self, data):
        """
        Method to actually select database and store data in there.

        Parameters:
            data: {"json", "sql", "nosql"}
            note: "json" saves additional table "id_table.json"
             with league names and related ids
        """
        if self.db_type == "json":
            self.store_json(data)
        elif self.db_type == "sql":
            self.store_sql(data)
        elif self.db_type == "nosql":
            self.store_nosql(data)
        else:
            raise AttributeError("invalid value for database type")

    def store_json(self, data):
        pass

    def store_sql(self, data):
        data = data[list(data.keys())[0]]
        for team, stats in data.items():
            for day_collected, stat in stats.items():
                # day_collected = datetime.datetime.strptime(day_collected, '%Y-%m-%d')
                in_stat_add = {}
                for stat_key, stat_value in stat.items():
                    if 'past' in stat_key or 'fut' in stat_key:
                        self.add_row_in_matches(stat_value, team)
                    else:
                        if stat_key == 'parts':
                            v = re.split(r',\s', stat_value)
                            v = [re.sub(r',', '.', i) for i in v]
                            v = [i.split(':') for i in v]
                            v = [[i[0], i[1][1:]] for i in v]
                            for i in v:
                                in_stat_add.update({i[0]: i[1]})
                        elif stat_key == 'scored_missed':
                            in_stat_add.update({'scored1': stat_value[0]})
                            in_stat_add.update({'missed1': stat_value[1]})
                        elif stat_key == 'wdl':
                            in_stat_add.update({'win1': stat_value[0]})
                            in_stat_add.update({'draw1': stat_value[1]})
                            in_stat_add.update({'lose1': stat_value[2]})
                        elif len(stat_value) == 1:
                            in_stat_add.update({stat_key: stat_value})
                        elif len(stat_value) == 2 and stat_value[1] == '':
                            in_stat_add.update({stat_key: stat_value[0]})
                        elif len(stat_value) == 2:
                            in_stat_add.update({stat_key: stat_value[1]})
                self.add_row_in_stat(in_stat_add)

class DatabaseSQL(Database):
    def __init__(self):

class Parser():
    def __init__(self, proxy=""):
        self.base_page = "https://nb-bet.com/Teams/121-Arsenal-statistika-komandi"

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--proxy-server=%s' % proxy)
        self.driver = webdriver.Chrome(chrome_options=chrome_options)


class NbBetParser(object):
    def __init__(self, proxy):
        self.home_page = 'https://nb-bet.com'
        self.proxy_list = ProxyList(PROXY)
        self.driver = self.initialize_driver(proxy)
        self.initial = 'https://nb-bet.com/Teams/121-Arsenal-statistika-komandi'
        self.date = str(datetime.datetime.now()).split(' ')[0]
        self.start_parse_date = str(datetime.datetime.now())
        self.current_team = None
        self.current_league = None
        self.change_period_to = None
        self.save_format = 'json'
        self.sql_db = None
        self.id_table = None
        self.cur = None
        self.matches_table_exists = False
        self.stat_table_exists = False
        self.n_columns_schedule = 1
        self.nosql_db = None
        self.nosql_coll = None
        self.is_checklist_full = False
        self.clear_checklist = True
        self.previous_list_of_teams = None

    def connect_to_db(self):
        if self.save_format == 'sql':
            self.sql_db = sql_db.connect(conn_info)
            self.cur = self.sql_db.cursor()
            # saving id_table
            id_table_create = """CREATE TABLE IF NOT EXISTS id_table (name VARCHAR(50), id INT PRIMARY KEY);"""
            self.cur.execute(id_table_create)

        if self.save_format == 'nosql':
            client = MongoClient()
            self.nosql_db = client['faiabolll']
            self.nosql_coll = self.nosql_db['football']

    def switch_driver_proxy(self, msg):
        print(msg)
        self.driver.quit()
        proxy = self.proxy_list.get_random_proxy()
        self.driver = self.initialize_driver(proxy)
        self.driver.get(self.initial)
        print(f'CONNECTED TO {proxy}')

    def initialize_driver(self, proxy):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--proxy-server=%s' % proxy)
        return webdriver.Chrome(chrome_options=chrome_options)

    def start_parse(self, save_format='json'):
        self.save_format = save_format

        try:
            self.driver.get(self.initial)
        except:
            print('Page is not exist')
            return

        self.connect_to_db()

        try:
            league_selector = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "MainContent_ddlLeagueHeaderGo")))
        except Exception as e:
            self.switch_driver_proxy('bad driver with probably bad prxoy\nswitching.............')

        list_of_leagues = league_selector.find_elements_by_tag_name('option')
        n_of_leagues = [(league.text, league.get_attribute('value')) for league in list_of_leagues]
        n_of_leagues = n_of_leagues
        # saving table of names of leagues and their id
        self.id_table = {name[0]: i for i, name in enumerate(n_of_leagues)}
        N = len(n_of_leagues)

        # clean file out of any symbols in start of parsing
        if self.clear_checklist:
            with open('checklist.txt', 'w') as f:
                f.write('')
            self.clear_checklist = False

        while not self.is_checklist_full:
            for i, n in enumerate(n_of_leagues):
                # switching proxy in specific case
                if i % 10 == 0 and i != 0:
                    self.switch_driver_proxy('switching proxy because of loop')

                # finding option element to click with waiting
                with open('checklist.txt', 'r') as f:
                    rows = f.readlines()
                # if this league is already in checklist program skip it
                if n[0] + '\n' in rows:
                    continue

                # finding element with a link to other league
                try:
                    option = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.XPATH, f'//option[@value={n[1]}]')))  # n[1] is id value to locate link element
                except:
                    # in case there is empty page it is necessary to switch proxy
                    try:
                        self.driver.find_element_by_class_name('vertical-align-middle-with-margin-logo')
                    except:
                        self.switch_driver_proxy('switching proxy because of unavailability to locate logo')
                    # in case there is no such element program goes back to previous page where everythiing works ok
                    self.driver.back()
                    continue
                option.click()
                self.wait_to_click(3)

                # league's parsing
                try:
                    results = self.parse_league()
                    self.current_league = n[0]
                    league_name_to_save = re.sub(r'\.', '_', self.current_league)
                    self.save_league(results, league_name_to_save)
                except IndexError as e:
                    alert_meassage = f"{e}\n\ncurrent index {i} out of {len(n_of_leagues)}\n{self.current_team}, {self.current_league}\n" + \
                                     f"{self.start_parse_date} started \n{str(datetime.datetime.now())} finished"
                    self.bot_send_message(alert_meassage)
                    print(f"""
						Parsing stopped because of exception: {e}
						started at {self.start_parse_date}
						finished at {str(datetime.datetime.now())}
						""")
                    break
                except NoSuchElementException:
                    no_elem_ex = f'troubles in {self.current_league}'
                    print(no_elem_ex)
                    self.bot_send_message(no_elem_ex)
                except Exception as e:  # any other untrackable exception
                    msg = e
                    print(msg)
                    self.bot_send_message(msg)
                else:
                    # it is to make a program continue job while loop apparantley is out of range
                    with open('checklist.txt', 'a') as f:
                        f.write(n[0] + '\n')
                    with open('checklist.txt', 'r') as f:
                        rows_of_checklist = f.readlines()
                    # and finish a job when all leagues will be parsed
                    if len(rows_of_checklist) == N:
                        self.is_checklist_full = True
                        # state = parser.start_parse()
                        return 'parsing is done'

    def save_league(self, to_save, league_name_to_save):
        if self.save_format == 'json':
            if 'id_table.json' not in os.listdir(os.getcwd()):
                with open('id_table.json', 'w') as f:
                    self.dump = json.dump(self.id_table, f)
            path = 'l{}_{}.json'.format(self.id_table[self.current_league], self.date)
            path = os.path.join('data', path)
            with open(path, 'w') as f:
                print(f'{path} saved...')
                json.dump(to_save, f)

        elif self.save_format == 'sql':
            to_save = to_save[list(to_save.keys())[0]]
            for team, stats in to_save.items():
                for day_collected, stat in stats.items():
                    # day_collected = datetime.datetime.strptime(day_collected, '%Y-%m-%d')
                    in_stat_add = {}
                    for stat_key, stat_value in stat.items():
                        if 'past' in stat_key or 'fut' in stat_key:
                            self.add_row_in_matches(stat_value, team)
                        else:
                            if stat_key == 'parts':
                                v = re.split(r',\s', stat_value)
                                v = [re.sub(r',', '.', i) for i in v]
                                v = [i.split(':') for i in v]
                                v = [[i[0], i[1][1:]] for i in v]
                                for i in v:
                                    in_stat_add.update({i[0]: i[1]})
                            elif stat_key == 'scored_missed':
                                in_stat_add.update({'scored1': stat_value[0]})
                                in_stat_add.update({'missed1': stat_value[1]})
                            elif stat_key == 'wdl':
                                in_stat_add.update({'win1': stat_value[0]})
                                in_stat_add.update({'draw1': stat_value[1]})
                                in_stat_add.update({'lose1': stat_value[2]})
                            elif len(stat_value) == 1:
                                in_stat_add.update({stat_key: stat_value})
                            elif len(stat_value) == 2 and stat_value[1] == '':
                                in_stat_add.update({stat_key: stat_value[0]})
                            elif len(stat_value) == 2:
                                in_stat_add.update({stat_key: stat_value[1]})
                    self.add_row_in_stat(in_stat_add)

        elif self.save_format == 'nosql':
            # key = list(to_save.keys())[0]
            # dict_to_save = {league_name_to_save:to_save[key]}
            dict_to_save = {league_name_to_save: to_save}
            saved = True
            # give it 10 tries if it can save with first time
            for i in range(10):
                try:
                    self.nosql_coll.insert_one(dict_to_save)
                except Exception as e:
                    print(f'Happened {e} in nosql saving; waitting 3 secs')
                    self.wait_to_click(3)
                else:
                    break
                if i == 9:
                    saved = False
            if not saved:
                self.save_format = 'json'
                self.save_league(to_save, league_name_to_save)
                self.save_format = 'nosql'

    def parse_league(self, recursion=0):
        team_selector = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "MainContent_ddlMatchHeaderGo")))

        list_of_teams = team_selector.find_elements_by_tag_name('option')
        list_of_teams = [team.get_attribute('value') for team in list_of_teams if
                         'statistika-komandi' in team.get_attribute('value')[1:]]

        league_results = {}
        if recursion == 10:
            msg = f'{self.current_league} is unavailable'
            print(msg);
            self.bot_send_message(msg)
        elif list_of_teams == self.previous_list_of_teams:
            self.wait_to_click(5)
            recursion += 1
            league_results = self.parse_league(recursion)
        else:
            for team in list_of_teams:
                try:
                    link = self.home_page + '/' + team
                    self.driver.get(link)
                    team_stats = self.parse_team()  # return json-like dict
                    name_to_save = team.split('-')[-3]
                    league_results.update({name_to_save: team_stats})
                except Exception as e:
                    msg = f'{e} occured in parse_league()\n{link}'
                    print(msg)
                    self.bot_send_message(msg)

        self.previous_list_of_teams = list_of_teams

        return league_results

    def parse_team(self):
        if self.change_period_to:
            self.change_period(self.change_period_to)

        date_of_stats = {}
        try:
            self.wait_to_click(0.5, 0.05)
            title = self.parse_title()
            stats = self.parse_stats()
            self.wait_to_click()
            series = self.parse_series()
            self.wait_to_click(0.5, 0.05)
            past_results = self.parse_past_results()
            self.wait_to_click(0.5, 0.05)
            schedule = self.parse_schedule()
            time = str(datetime.datetime.now())
            print(self.current_team, 'from', self.current_league, 'parsed at',
                  time)  # self.current_team is determined in parse_title
            all_stats = {**title, **stats, **series, **past_results, **schedule}
            date_of_stats = {self.date: all_stats}
        except Exception as e:
            msg = f'{e} occured in parse_team()\n{self.current_team} from {self.current_league}'
            print(msg)
            self.bot_send_message(msg)
        return {self.current_team: date_of_stats}

    def parse_title(self):
        self.current_team = self.driver.find_element_by_xpath("//div[@id='MainContent_pnlContent']/h2").text
        position = self.driver.find_element_by_xpath("//div[@id='MainContent_pnlContent']/div[4]/span").text
        position = position.split(' ')[-1]
        WDL = self.driver.find_element_by_xpath("//div[@id='MainContent_pnlContent']/div[5]")
        WDL = WDL.find_elements_by_tag_name("strong")
        WDL = [n.text for n in WDL]
        scored_missed = self.driver.find_element_by_xpath("//div[@id='MainContent_pnlContent']/div[6]/span")
        scored_missed = scored_missed.find_elements_by_tag_name("strong")
        scored_missed = [sm.text for sm in scored_missed[-2:]]
        parts = self.driver.find_element_by_xpath("//div[@id='MainContent_pnlContent']/div[7]").text
        return {'wdl': WDL, 'scored_missed': scored_missed, 'parts': parts}

    def parse_stats(self):

        stats = {}
        table_cell = self.driver.find_element_by_id("MainContent_TabContainer_body")

        for cell in table_cell.find_elements_by_tag_name('table'):
            cells1 = cell.find_elements_by_class_name("tournaments-stats-cells1")
            for tr in cells1:
                tds = tr.find_elements_by_tag_name('td')
                tds = list(map(lambda x: re.sub(r'\.', '_', x.text), tds))
                if ''.join(tds) != '':
                    stats.update({tds[0]: (tds[1], tds[2])})
            cells2 = cell.find_elements_by_class_name("tournaments-stats-cells2")
            for tr in cells2:
                tds = tr.find_elements_by_tag_name('td')
                tds = list(map(lambda x: re.sub(r'\.', '_', x.text), tds))
                if ''.join(tds) != '':
                    stats.update({tds[0]: (tds[1], tds[2])})
        return stats

    def parse_series(self):
        series = {}
        series_button = self.driver.find_element_by_id("MainContent_TabContainer_ctl03_tab")
        series_button.click()
        tabel_container = self.driver.find_element_by_id("MainContent_TabContainer_ctl03")
        trs1 = tabel_container.find_elements_by_class_name("tournaments-stats-cells1")
        trs2 = tabel_container.find_elements_by_class_name("tournaments-stats-cells2")
        for tr in trs1:
            tds = tr.find_elements_by_tag_name('td')
            tds = list(map(lambda x: re.sub(r'\.', '_', x.text), tds))
            series.update({tds[0] + 'left': tds[1]})
        for tr in trs2:
            tds = tr.find_elements_by_tag_name('td')
            tds = list(map(lambda x: re.sub(r'\.', '_', x.text), tds))
            series.update({tds[0] + 'right': tds[1]})

        return series

    def parse_past_results(self):
        results = {}
        results_button = self.driver.find_element_by_id("__tab_MainContent_TabContainer_tpResults")
        results_button.click()

        tabel_container = self.driver.find_element_by_id("MainContent_TabContainer_tpResults")
        trs = tabel_container.find_elements_by_class_name("rows-border-top")
        for i, tr in enumerate(trs):
            tds = tr.find_elements_by_tag_name('td')
            extract_td_data = lambda x: x.text
            tr_info = list(map(extract_td_data, tds))
            if self.save_format != 'json':
                y = re.findall(r'\.([1-2][0-9])\s', tr_info[0])[0]
                tr_info[0] = re.sub(r'\.[1-2][0-9]\s', '.20' + y + ' ', tr_info[0])
                tr_info[0] = datetime.datetime.strptime(tr_info[0], '%d.%m.%Y %H:%M')
            results.update({'past' + str(i): tr_info})

        return results

    def parse_schedule(self):
        schedule = {}
        try:
            schedule_button = self.driver.find_element_by_id("__tab_MainContent_TabContainer_tpSchedule")
            schedule_button.click()
        except:
            print('No schedule for', self.current_team)
            # assuming situation that first leagues and teams parsed succefully and created tables
            schedule = {'n' + str(i): 'NULL' for i in range(self.n_columns_schedule)}
        else:
            tabel_container = self.driver.find_element_by_id("MainContent_TabContainer_tpSchedule")
            trs = tabel_container.find_elements_by_class_name("rows-border-top")
            for i, tr in enumerate(trs):
                tds = tr.find_elements_by_tag_name('td')
                extract_td_data = lambda x: x.text
                tr_info = list(map(extract_td_data, tds))
                # processing matchday timestamp in case it is not json format

                if self.save_format != 'json':
                    y = re.findall(r'\.([1-2][0-9])\s', tr_info[0])[0]
                    tr_info[0] = re.sub(r'\.[1-2][0-9]\s', '.20' + y + ' ', tr_info[0])
                    tr_info[0] = datetime.datetime.strptime(tr_info[0], '%d.%m.%Y %H:%M')
                schedule.update({'fut' + str(i): tr_info})
                # for no-schedule-exception
                self.n_columns_schedule = len(tr_info)
        return schedule

    def add_row_in_stat(self, row):
        row = [[k, v] for k, v in row.items()]
        if not self.stat_table_exists:
            try:
                delete_query = """DROP TABLE stat_table"""
                self.cur.execute(delete_query)
                self.sql_db.commit()
            except:
                self.sql_db.rollback()
                trans = lambda x: re.sub(r'\s|\)|\(|\+|\.|\-|\<', '_',
                                         translator.translate(x, src='ru', dest='en').text) + ' TEXT'
                dtypes = [col[0] for col in row]
                dtypes = ['day_collected TEXT'] + list(map(trans, dtypes))
                dtypes = '(' + ', '.join(dtypes) + ')'
                stat_table_create = "CREATE TABLE IF NOT EXISTS stat_table " + dtypes
                self.cur.execute(stat_table_create)
                self.sql_db.commit()
        values = str(tuple([str(self.date)] + [col[1] for col in row]))
        add_query = """INSERT INTO stat_table VALUES %s""" % values
        self.cur.execute(add_query)
        self.sql_db.commit()

    def add_row_in_matches(self, row, team_name):

        if not self.matches_table_exists:
            # matches_table_create = """CREATE TABLE IF NOT EXISTS matches_table (
            # day_collected VARCHAR(10), match_date VARCHAR(19), league VARCHAR(50), team VARCHAR(30), opponent VARCHAR(30),
            # is_forecast VARCHAR(7), match_score VARCHAR(7), first_score VARCHAR(7), match_result VARCHAR(3));"""
            try:
                delete_query = """DROP TABLE matches_table"""
                self.cur.execute(delete_query)
            except:
                self.sql_db.rollback()
                matches_table_create = """CREATE TABLE IF NOT EXISTS matches_table (
				day_collected TEXT, match_date TEXT, league TEXT, team TEXT, opponent TEXT,
				is_forecast TEXT, match_score TEXT, first_score TEXT, match_result TEXT);"""
                self.cur.execute(matches_table_create)

        match_score = 'NULL'
        first_score = 'NULL'
        values_to_add = ['NULL' if x == '' else x for x in row]
        is_forecast = True if 'ПС' in values_to_add[-2] else False
        match_result = values_to_add[-1]
        opponent = re.sub(team_name, '', values_to_add[2])
        opponent = re.sub(r' - ', '', opponent)

        values = [self.date, values_to_add[0], values_to_add[1], team_name, opponent,
                  is_forecast, match_score, first_score, match_result]
        # in case saving timestamps
        # values=[self.date, values_to_add[0]]+[str(i) for i in values[2:]]
        # values = ', '.join(values)
        values = str(tuple([str(i) for i in values]))

        if values_to_add[-2] != 'NULL':
            if is_forecast:
                match_score = ''.join(values_to_add[-2].split(' ')[-3:])
                first_score = 'NULL'
            else:
                temp = values_to_add[-2].split('(')
                match_score = re.sub(r'\s', '', temp[0])
                first_score = re.sub(r'\s', '', temp[1][:-1])

        add_query = """INSERT INTO matches_table VALUES %s""" % values
        self.cur.execute(add_query)
        self.sql_db.commit()

    def bot_send_message(self, msg):
        if TG_CONNECTION:
            try:
                bot.send_message(ME, msg)
            except Exception as e:
                print(e)
        else:
            print('Connection to bot is unavailable')

    def wait_to_click(self, m=2, s=0.2):
        sleep(random.gauss(m, s))

    def change_period(self):
        self.wait_to_click(3)
        chain = ActionChains(self.driver)
        period_element = self.driver.find_element_by_name("ctl00$MainContent$txtNum")
        refresh_button = self.driver.find_element_by_class_name("btn-blue-reminder")
        chain.move_to_element(period_element)
        chain.double_click(period_element)
        chain.send_keys(u'\ue017')
        chain.send_keys(str(period))
        chain.click(refresh_button)
        chain.perform()


def main():
    parser = NbBetParser(PROXY)
    state = parser.start_parse(save_format='nosql')
    print(state)


if __name__ == '__main__':
    main()
