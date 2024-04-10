from webmd.items import WebmdItem
from scrapy import Spider, Request
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
import urllib
import re
import html

TCA = ['amitriptyline', 'clomipramine', 'coxepin', 'cortriptyline', 'imipramine', 'dosulepin']
MAOI = ['tranylcypromine', 'moclobemide', 'phenelzine', 'selegiline', 'isocarboxazid']
SSRI = ['fluoxetine', 'paroxetine', 'sertraline', 'citalopram', 'escitalopram', 'fluvoxamine']
SNRI = ['venlafaxine', 'duloxetine', 'desvenlafaxine', 'levomilnacipran', 'milnacipran']
Benzodiazepines = ['temazepam', 'nitrazepam', 'diazepam', 'oxazepam', 'alprazolam', 'lorazepam']
AtypicalAntipsychotics = ['aripiprazole', 'olanzapine', 'quetiapine', 'risperidone', 'ziprasidone', 'clozapine']
GABA = ['gabapentin', 'pregabalin', 'tiagabine', 'vigabatrin', 'valproate', 'carbamazepine']
MixedAntidepressants = ['bupropion', 'mirtazapine', 'trazodone']

Drugs = TCA + MAOI + SSRI + SNRI + Benzodiazepines + AtypicalAntipsychotics + GABA + MixedAntidepressants
DrugsDict = {}
for d in TCA:
    DrugsDict[d] = 'TCA'
for d in MAOI:
    DrugsDict[d] = 'MAOI'
for d in SSRI: 
    DrugsDict[d] = 'SSRI'
for d in SNRI:
    DrugsDict[d] = 'SNRI'
for d in Benzodiazepines:
    DrugsDict[d] = 'Benzodiazepines'
for d in AtypicalAntipsychotics:
    DrugsDict[d] = 'AtypicalAntipsychotics'
for d in GABA:
    DrugsDict[d] = 'GABA'
for d in MixedAntidepressants:
    DrugsDict[d] = 'MixedAntidepressants'

headers = {'User-Agent': 'Chrome/60.0.3112.113', 
           'enc_data': 'OXYIMo2UzzqFUzYszFv4lWP6aDP0r+h4AOC2fYVQIl8=', 
           'timestamp': 'Mon, 04 Sept 2017 04:35:00 GMT', 
           'client_id': '3454df96-c7a5-47bb-a74e-890fb3c30a0d'}

# Check if the drug is in the list of DrugsDict, based on Drug and GenName
def check_if_drug_in_list(Drug, GenName):
    for d in DrugsDict.keys():
        if d in Drug or d in GenName:
            return True
    return False
            
class WebmdSpider(Spider):
    
    name = "webmd"
    allowed_urls = ['http://www.webmd.com/']
    start_urls = ['http://www.webmd.com/drugs/2/index']
    drug_dict = {}

    def parse(self, response):
        yield Request(response.urljoin('/drugs/2/conditions/index'),
                      callback = self.parse_conditions,
                      dont_filter = True)
        
        #for i in range(len(atoz)):
        #    yield Request(response.urljoin(atoz[i]), 
        #                  callback = self.parse_sub, 
        #                  dont_filter = True)
            
    def parse_sub(self, response):
        sub = response.xpath('//ul[@class="browse-letters squares sub-alpha sub-alpha-letters"]')[0].xpath("li/a/@href").extract()
        for i in range(len(sub)):
            yield Request(response.urljoin(sub[i]), 
                          callback = self.parse_drug, 
                          dont_filter = True)
    
    def parse_drug(self, response):
        try:
            drug_list = response.xpath('//div[@class="drugs-search-list-conditions"]/ul')[0].xpath("li/a")
        except:
            drug_list = []

        for i in range(len(drug_list)):
            yield Request(response.urljoin(drug_list[i].xpath("@href")[0].extract()),
                          callback = self.parse_details, 
                          meta = {'Drug': drug_list[i].xpath("text()")[0].extract().lower()},
                          dont_filter = True)    

    def parse_details(self, response):
        print('parse_details start')
        Use = ' '.join(response.xpath(
            '//div[contains(concat(" ", normalize-space(@class), " "), " uses-container ")]/div[2]//text()'
            ).extract())
        print('parse_details Use', Use, 'Use end')

        HowtoUse = ' '.join(response.xpath(
            '//div[contains(concat(" ", normalize-space(@class), " "), " how-to-use-section ")]/div//text()'
            ).extract())
        print('parse_details HowtoUse', HowtoUse, 'HowtoUse end')

        Sides = ' '.join(response.xpath(
            '//div[contains(concat(" ", normalize-space(@class), " "), " side-effects-container ")]/div[3]//text()'
            ).extract())
        print('parse_details Sides', Sides, 'Sides end')

        Precautions = ' '.join(response.xpath(
            '//div[contains(concat(" ", normalize-space(@class), " "), " precautions-container ")]/div[3]//text()'
            ).extract())
        print('parse_details Precautions', Precautions, 'Precautions end')

        Interactions = ' '.join(response.xpath(
            '//div[contains(concat(" ", normalize-space(@class), " "), " interactions-container ")]/div[3]//text()'
            ).extract())
        print('parse_details Interactions', Interactions, 'Interactions end')

        # tRevurl is the url to the reviews page
        tRevurl = response.xpath('//ul[@class="auto-tabs"]/li/a[contains(text(), "Reviews")]/@href')

        if len(tRevurl) is not 0:
            revurl = tRevurl.extract()[0]    
            if not Use:
                Use = ' '
            if not Sides:
                Sides = ' '
            if not Interactions:
                Interactions = ' '
            if not Precautions:
                Precautions = ' '
            if not HowtoUse:
                HowtoUse = ' '
                
            if re.search('common brand', response.body.decode('utf-8').lower()):
                Brand = response.xpath('//h3[@class="drug-generic-name"][1]/a/text()')
                BrandName = Brand[0].extract() if Brand else ' '
                Gen = response.xpath('//h3[@class="drug-generic-name"][2]/a/text()')
                GenName = Gen[0].extract() if Gen else ' '
            elif re.search('generic name', response.body.decode('utf-8').lower()):
                BrandName = ' '
                Gen = response.xpath('//h3[@class="drug-generic-name"][1]/a/text()')
                GenName = Gen[0].extract() if Gen else ' '
            else:
                GenName = ' '
                BrandName = ' '

            print('parse_details revurl', revurl, 'revurl end')
            print('parse_details BrandName', BrandName, 'BrandName end')
            print('parse_details GenName', GenName, 'GenName end')
            print('url contra', response.urljoin(response.url + '/list-contraindications'))

            if check_if_drug_in_list(response.meta['Drug'], GenName):
            # If the drug is not in the list of drugs we want to scrape, we don't scrape it
                yield Request(response.urljoin(response.url + '/list-contraindications'),
                            callback = self.parse_avoid,
                            meta = {'Drug': response.meta['Drug'], 
                                    'Use': Use,
                                    'HowtoUse': HowtoUse,
                                    'Sides': Sides,
                                    'Precautions': Precautions,
                                    'Interactions': Interactions,
                                    'revurl': revurl,
                                    'BrandName': BrandName,
                                    'GenName': GenName,
                                    'Condition': response.meta['Condition'],
                                    'Indication': response.meta['Indication'],
                                    'Type': response.meta['Type']},
                            dont_filter = True)
            else:
                print('Drug not in list', response.meta['Drug'])

    def parse_avoid(self, response):
        avoid_use_match = re.findall(r'Conditions:|We\'re sorry, but we couldn\'t find the page you tried', response.body.decode('utf-8'))
        allergies_match = re.findall(r'Allergies:', response.body.decode('utf-8'))

        if 'We\'re sorry, but we couldn\'t find the page you tried' in avoid_use_match:
            AvoidUse = ' '
            Allergies = ' '
        elif 'Conditions:' in avoid_use_match:
            AvoidUse = ' '.join(response.xpath('//*[@id="ContentPane28"]/div/article/section/p[2]/text()').extract())
            Allergies = ' '.join(response.xpath('//*[@id="ContentPane28"]/div/article/section/p[3]/text()').extract())
        elif 'Allergies:' in allergies_match:
            AvoidUse = ' '
            Allergies = ' '.join(response.xpath('//*[@id="ContentPane28"]/div/article/section/p[2]/text()').extract())
        else:
            AvoidUse = ' '
            Allergies = ' '

        AvoidUse = AvoidUse if AvoidUse else ' '
        Allergies = Allergies if Allergies else ' '

        print('parse_avoid AvoidUse', AvoidUse, 'AvoidUse end')
        print('parse_avoid Allergies', Allergies, 'Allergies end')
        print(avoid_use_match, allergies_match)

        print('revurl', response.meta['revurl'])

        if response.meta['revurl'] == 'javascript:void()':
            WebmdSpider.drug_dict[response.meta['Drug']] = {'Use': response.meta['Use'],
                                                            'HowtoUse': response.meta['HowtoUse'],
                                                            'Sides': response.meta['Sides'],
                                                            'Precautions': response.meta['Precautions'],
                                                            'Interactions': response.meta['Interactions'],
                                                            'BrandName': response.meta['BrandName'],
                                                            'GenName': response.meta['GenName'],
                                                            'AvoidUse': AvoidUse,
                                                            'Allergies': Allergies,
                                                            'Effectiveness': ' ',
                                                            'EaseOfUse': ' ',
                                                            'Satisfaction': ' ',
                                                            'Reviews': [{}]}
        else:
            yield Request(response.urljoin(response.meta['revurl']),
                        callback = self.parse_reviews,
                        meta = {'Drug': response.meta['Drug'], 
                                'Use': response.meta['Use'],
                                'HowtoUse': response.meta['HowtoUse'],
                                'Sides': response.meta['Sides'],
                                'Precautions': response.meta['Precautions'],
                                'Interactions': response.meta['Interactions'],
                                'BrandName': response.meta['BrandName'],
                                'GenName': response.meta['GenName'],
                                'Condition': response.meta['Condition'],
                                'Indication': response.meta['Indication'],
                                'Type': response.meta['Type'],
                                'AvoidUse': AvoidUse,
                                'Allergies': Allergies},
                        dont_filter = True)

    def parse_reviews(self, response):
        if re.search('Rate this treatment and share your opinion', response.body.decode('utf-8')) \
           or re.search('Be the first to share your experience with this treatment', response.body.decode('utf-8')):
            WebmdSpider.drug_dict[response.meta['Drug']] = {'Use': response.meta['Use'],
                                                            'HowtoUse': response.meta['HowtoUse'],
                                                            'Sides': response.meta['Sides'],
                                                            'Precautions': response.meta['Precautions'],
                                                            'Interactions': response.meta['Interactions'],
                                                            'BrandName': response.meta['BrandName'],
                                                            'GenName': response.meta['GenName'],
                                                            'AvoidUse': response.meta['AvoidUse'],
                                                            'Allergies': response.meta['Allergies'],
                                                            'Effectiveness': ' ',
                                                            'EaseOfUse': ' ',
                                                            'Satisfaction': ' ',
                                                            'Reviews': [{}]}

        else:
            # drugid for the drug you want rating summary for
            # secondaryId for the specific condition of reviewers (-1 for overall reviews)
            # secondaryIdValue seems to not affect anything?
            NumReviews = int(''.join(filter(str.isdigit, response.xpath('//li[@class="active-tab"]/a/span/text()')[0].extract())))  
            print('Number of reviews', NumReviews)
            url = 'http://www.webmd.com/drugs/service/UserRatingService.asmx/GetUserReviewSummary?repositoryId=1&primaryId='
            DrugId = re.search('(drugid=)(\d+)', response.body.decode('utf-8').lower()).group(2)
            url2 = '&secondaryId=-1&secondaryIdValue='
            id2 = '0' #urllib.parse.quote(re.sub("\s+", " ", response.xpath('//option[@value = -1]//text()').extract()[0]).strip())
            print(url + DrugId + url2 + id2)

            yield Request(url + DrugId + url2 + id2,
                          callback = self.parse_ratings,
                          meta = {'Drug': response.meta['Drug'], 
                                  'Use': response.meta['Use'],
                                  'HowtoUse': response.meta['HowtoUse'],
                                  'Sides': response.meta['Sides'],
                                  'Precautions': response.meta['Precautions'],
                                  'Interactions': response.meta['Interactions'],
                                  'BrandName': response.meta['BrandName'],
                                  'GenName': response.meta['GenName'],
                                  'AvoidUse': response.meta['AvoidUse'],
                                  'Allergies': response.meta['Allergies'],
                                  'Condition': response.meta['Condition'],
                                  'Indication': response.meta['Indication'],
                                  'Type': response.meta['Type'],
                                  'DrugId': DrugId,
                                  'NumReviews': NumReviews},
                          dont_filter = True)
                
    def parse_ratings(self, response):
        values = []
        for i in range(3, 6):
            try:
                matches = re.findall('("xsd:string">)(\d+\.?\d*)', response.xpath('//*/*').extract()[i])
                value = matches[0][1] if matches else ' '
            except IndexError:
                value = ' '
            values.append(value)

        Effectiveness, EaseofUse, Satisfaction = values

        print('parse_ratings Effectiveness', Effectiveness, 'Effectiveness end')
        print('parse_ratings EaseofUse', EaseofUse, 'EaseofUse end')
        print('parse_ratings Satisfaction', Satisfaction, 'Satisfaction end')

        url = "http://www.webmd.com/drugs/service/UserRatingService.asmx/GetUserReviewsPagedXml?repositoryId=1&objectId="
        url2 = "&pageIndex=0&pageSize="
        url3 = "&sortBy=DatePosted"
        print(url + response.meta['DrugId'] + url2 + str(response.meta['NumReviews']) + url3)
        yield Request(url + response.meta['DrugId'] + url2 + str(response.meta['NumReviews']) + url3,
                      method = 'GET', headers=headers,
                      callback = self.parse_all_reviews,
                      meta = {'Drug': response.meta['Drug'], 
                              'Use': response.meta['Use'],
                              'HowtoUse': response.meta['HowtoUse'],
                              'Sides': response.meta['Sides'],
                              'Precautions': response.meta['Precautions'],
                              'Interactions': response.meta['Interactions'],
                              'BrandName': response.meta['BrandName'],
                              'GenName': response.meta['GenName'],
                              'AvoidUse': response.meta['AvoidUse'],
                              'Allergies': response.meta['Allergies'],
                              'DrugId': response.meta['DrugId'],
                              'NumReviews': response.meta['NumReviews'],
                              'Condition': response.meta['Condition'],
                              'Indication': response.meta['Indication'],
                              'Type': response.meta['Type'],
                              'Effectiveness': Effectiveness,
                              'EaseofUse': EaseofUse,
                              'Satisfaction': Satisfaction},
                      dont_filter = True)
        
    def parse_all_reviews(self, response):
        print('parse_all_reviews start', response.url)
        n = response.meta['NumReviews']
        data = Selector(text=html.unescape(response.xpath("//*")[0].extract()).replace("<![CDATA[", "").replace("]]>", ""))
        Reviews = [{} for i in range(int(n))]
        print(n)
        for i in range(int(n)):
            try:
                t_Id = data.xpath("//userreviewid")[i].xpath("text()")
                Reviews[i]['Id'] = ' ' if len(t_Id) is 0 else t_Id[0].extract()
                t_Condition = data.xpath("//secondaryvalue")[i].xpath("text()")
                Reviews[i]['Condition'] = ' ' if len(t_Condition) is 0 else t_Condition[0].extract()
                t_IsPatient = data.xpath("//boolean2")[i].xpath("text()")
                Reviews[i]['IsPatient'] = ' ' if len(t_IsPatient) is 0 else t_IsPatient[0].extract()
                t_IsMale = data.xpath("//boolean1")[i].xpath("text()")
                Reviews[i]['IsMale'] = ' ' if len(t_IsMale) is 0 else t_IsMale[0].extract()
                t_Age = data.xpath("//lookuptext1")[i].xpath("text()")
                Reviews[i]['Age'] = ' ' if len(t_Age) is 0 else t_Age[0].extract()
                t_TimeUsingDrug = data.xpath("//lookuptext2")[i].xpath("text()")
                Reviews[i]['TimeUsingDrug'] = ' ' if len(t_TimeUsingDrug) is 0 else t_TimeUsingDrug[0].extract()
                t_DatePosted = data.xpath("//dateposted")[i].xpath("text()")
                Reviews[i]['DatePosted'] = ' ' if len(t_DatePosted) is 0 else t_DatePosted[0].extract()
                t_Comment = data.xpath("//userexperience")[i].xpath("text()")
                Reviews[i]['Comment'] = ' ' if len(t_Comment) is 0 else t_Comment[0].extract()
                t_Effectiveness = data.xpath("//ratingcriteria1")[i].xpath("text()")
                Reviews[i]['Effectiveness'] = ' ' if len(t_Effectiveness) is 0 else t_Effectiveness[0].extract()
                t_EaseOfUse = data.xpath("//ratingcriteria2")[i].xpath("text()")
                Reviews[i]['EaseOfUse'] = ' ' if len(t_EaseOfUse) is 0 else t_EaseOfUse[0].extract()
                t_Satisfaction = data.xpath("//ratingcriteria3")[i].xpath("text()")
                Reviews[i]['Satisfaction'] = ' ' if len(t_Satisfaction) is 0 else t_Satisfaction[0].extract()
                t_NumFoundHelpful = data.xpath("//foundhelpfulcount")[i].xpath("text()")
                Reviews[i]['NumFoundHelpful'] = ' ' if len(t_NumFoundHelpful) is 0 else t_NumFoundHelpful[0].extract()
                t_NumVoted = data.xpath("//totalvotedcount")[i].xpath("text()")
                Reviews[i]['NumVoted'] = ' ' if len(t_NumVoted) is 0 else t_NumVoted[0].extract()
            except IndexError:
                Reviews[i]['Id'] = ' '
                Reviews[i]['Condition'] = ' '
                Reviews[i]['IsPatient'] = ' '
                Reviews[i]['IsMale'] = ' '
                Reviews[i]['Age'] = ' ' 
                Reviews[i]['TimeUsingDrug'] = ' '
                Reviews[i]['DatePosted'] = ' '
                Reviews[i]['Comment'] = ' '
                Reviews[i]['Effectiveness'] = ' '
                Reviews[i]['EaseOfUse'] = ' '
                Reviews[i]['Satisfaction'] = ' '
                Reviews[i]['NumFoundHelpful'] = ' '
                Reviews[i]['NumVoted'] = ' '

        info = {'Drug': response.meta['Drug'],
                                                        'Use': response.meta['Use'],
                                                        'HowtoUse': response.meta['HowtoUse'],
                                                        'Sides': response.meta['Sides'],
                                                        'Precautions': response.meta['Precautions'],
                                                        'Interactions': response.meta['Interactions'],
                                                        'BrandName': response.meta['BrandName'],
                                                        'GenName': response.meta['GenName'],
                                                        'AvoidUse': response.meta['AvoidUse'],
                                                        'Allergies': response.meta['Allergies'],
                                                        'DrugId': response.meta['DrugId'],
                                                        'NumReviews': response.meta['NumReviews'],
                                                        'Effectiveness': response.meta['Effectiveness'],
                                                        'EaseofUse': response.meta['EaseofUse'],
                                                        'Satisfaction': response.meta['Satisfaction'],
                                                        'Condition': response.meta['Condition'],
                                                        'Indication': response.meta['Indication'],
                                                        'Type': response.meta['Type'],
                                                        'Reviews': Reviews} 
        # Set this so we can check if the drug has already been scraped
        WebmdSpider.drug_dict[response.meta['Drug']] = info
        item = WebmdItem()
        for key in info.keys():
            item[key] = info[key]
        print('yielding item', item)
        yield item
            
    def parse_conditions(self, response):
        atoz = response.xpath('//ul[@class="browse-letters squares"]')[0].xpath("li/a/@href").extract()
        for i in range(len(atoz)):
            print('parse_conditions', response.urljoin(atoz[i]))
            yield Request(response.urljoin(atoz[i]),
                          callback = self.parse_condition,
                          dont_filter = True)
            
    def parse_condition(self, response):
        drug_list = response.xpath('//div[@class="drugs-search-list-conditions"]/ul')[0].xpath("li/a")
        print('parse_condition', response.urljoin(drug_list[0].xpath("@href")[0].extract()))
        for i in range(len(drug_list)):
            yield Request(response.urljoin(drug_list[i].xpath("@href")[0].extract()),
                          callback = self.parse_condition_drug, 
                          meta = {'Condition' : drug_list[i].xpath("text()")[0].extract()}, 
                          dont_filter = True)
    
    def parse_condition_drug(self, response):
        print('parse_condition_drug', response.url)
        drugs = response.xpath("//div[@class='medication-results-list']")[0].xpath("div")

        for i in range(len(drugs)):
            try:
                Link = drugs[i].xpath("span[1]/a")[0].xpath("@href")[0].extract()
            except IndexError:
                Link = ' '
            try:
                Drug = drugs[i].xpath("span[1]/a/text()")[0].extract()
            except IndexError:
                Drug = ' '
            try:
                Indication = drugs[i].xpath("span[2]/text()")[0].extract()
            except IndexError:
                Indication = ' '
            try:
                Type = drugs[i].xpath("span[3]/text()")[0].extract()
            except IndexError:
                Type = ' '

            url = response.urljoin(Link)
            print('parse_condition_drug', Link, Drug, Indication, Type, url)

            yield Request(url,
                          callback = self.check_drug,
                          meta = {'Drug': Drug, 
                                  'Condition': response.meta['Condition'],
                                  'Indication': Indication,
                                  'Type': Type},
                          dont_filter = True)
            
    def check_drug(self, response):
        Drug = response.meta['Drug']
        if Drug in WebmdSpider.drug_dict.keys():
            print("OLD DRUG: " + Drug) 
            info = WebmdSpider.drug_dict[Drug]
            item = WebmdItem()
            item['Condition'] = response.meta['Condition']
            item['Indication'] = response.meta['Indication']
            item['Type'] = response.meta['Type']
            for key in info.keys():
                item[key] = info[key]
            yield item
        else:
            print("NEW DRUG: " + Drug) 
            yield Request(response.url,
                          callback = self.parse_details,
                          meta = {'Drug': Drug, 
                                  'Condition': response.meta['Condition'],
                                  'Indication': response.meta['Indication'],
                                  'Type': response.meta['Type']},
                          dont_filter = True)

                
                
                
                
                
'''print("NOT ANOMALY: " + Drug) 
                info = WebmdSpider.drug_dict[Drug]
                item = WebmdItem()
                item['Condition'] = response.meta['Condition']
                item['Drug'] = Drug
                item['Indication'] = Indication
                item['Type'] = Type
                for key in info.keys():
                    item[key] = info[key]
                yield item'''


# Test the git thing