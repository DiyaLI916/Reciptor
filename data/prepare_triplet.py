import json
import os
import random
query_similar_recipe = '''
prefix recipe-kb: <http://idea.rpi.edu/heals/kb/>
prefix spa: <tag:stardog:api:analytics:>

SELECT ?similarRecipeTitle ?confidence ?g
WHERE {
    GRAPH spa:model {
      :s4   spa:arguments (?ingredients) ;
            spa:confidence ?confidence ;
            spa:parameters [ spa:limit 15000 ] ;
            spa:predict ?similarRecipe .
    }

    GRAPH ?g { 
      ?similarRecipe rdfs:label ?similarRecipeTitle
    }
    {
        SELECT 
        ?recipe (spa:set(?ingre_new) as ?ingredients) 
        WHERE{
        GRAPH ?g { 
                 ?recipe  recipe-kb:uses ?ingre_name;
                          rdfs:label ?query_recipe_title .
                 }
        BIND(replace(str(?ingre_name), str(?recipe), "") as ?ingre_new)
             }
        GROUP BY ?recipe
    } 
}
ORDER BY DESC(?confidence)
LIMIT 500
'''

query_url = '''
prefix prov: <http://www.w3.org/ns/prov#>
SELECT DISTINCT ?g2 ?foodcom
where{
        GRAPH ?g2 { ?kbassertion prov:wasDerivedFrom ?foodcom }
    }
limit 5
'''
triplet_ls = []
foodcom = {}
url_ls = []
with open('../../Pycharm/im2recipe-Pytorch/data/recipe1M/foodcom_light.json') as json_file:
    data = json.load(json_file)
    for record in data:
        if record['url'] not in foodcom.keys():
            url_ls.append(record['url'])
            foodcom[record['url']] = {}
            foodcom[record['url']]['id'] = record['id']
            foodcom[record['url']]['title'] = record['title']
            foodcom[record['url']]['partition'] = record['partition']
        else:
            print("there are redundant urls")
            exit()

print(len(url_ls))

if os.path.isfile('remain_url.json'):
    with open('remain_url.json') as json_file:
        url_ls = json.load(json_file)
        print("load remain url", len(url_ls))
else:
    with open('remain_url.json', 'w') as out_file:
        json.dump(url_ls, out_file)

def get_sampleID(kb_assertion, anchor_url):
    # kb_assertion = "<http://idea.rpi.edu/heals/kb/assertions-8ce6d00204b4a4160cb9632bbab9a80728e178e3>"
    kb_assertion = '<' + kb_assertion + '>'

    # not working
    # stardog query -b s="<http://example.org/test>" p=ex:name o="\"John Doe\"" myDb "select * {?s ?p ?o}"
    # stardog_query = 'stardog query -b s=\"' + kb_assertion + '\" recipe \"' + query_url + '\"'

    query_url_specific = query_url.replace('?kbassertion', kb_assertion)

    stardog_query = 'stardog query recipe \"' + query_url_specific + '\"'
    # print('stardog_query is:\n', stardog_query)

    # no newline in stardog query execution
    stardog_query = stardog_query.replace('\n', ' ')
    # stardog_query = 'stardog query recipe 1select_tag.sparql'

    stream = os.popen(stardog_query)
    output = stream.read().strip()

    # print(output)

    output = output.split('\n')
    # print(len(output))

    # get most relevant recipe
    foodcom_url = ''
    for rank in range(3, len(output) - 3):
        # print('rank', output[rank])
        sim_recipe = output[rank].split('|')
        # print(sim_recipe)
        g2 = sim_recipe[1].strip()
        if 'http://idea.rpi.edu/heals/kb' not in g2:
            foodcom_url += sim_recipe[2].strip()
        else:
            foodcom_url = sim_recipe[2].strip()
        print('foodcom url', foodcom_url)

        if foodcom_url in url_ls:
            # print("****\nfind the pos/neg sample", foodcom_url, "of anchor", anchor_url, "\n****")
            print("****\nfind the pos/neg sample\n****")
            return foodcom_url
        # has problem
        if 'http://idea.rpi.edu/heals/kb' not in g2:
            foodcom_url = ''
    return False

def prepare_sparql_query():
    iteration = 0
    while len(url_ls) > 0:
        anchor_url = random.choice(url_ls)
        # print("random item from list is: ", anchor_url)
        print(foodcom[anchor_url])

        anchor_query = foodcom[anchor_url]['title']
        # anchor_query = "Lentil Walnut \"no Meat\" Loaf"
        # anchor_query = "Subru Uncle's Toor Ki Dal(sindhi Style) Dad, Mom and I Love And"
        if '\'' in anchor_query:
            anchor_query = anchor_query.replace('\'', '\\\\\'')
        if '\"' in anchor_query:
            anchor_query = anchor_query.replace('\"', '\\\\\\"')


        anchor_query_normal = "\"" + anchor_query + "\""
        query_similar_recipe2 = query_similar_recipe.replace('?query_recipe_title', anchor_query_normal)

        with open('query_similar_recipe.sparql', 'w') as f:
            f.write(query_similar_recipe2)
        stardog_query = 'stardog query recipe query_similar_recipe.sparql'
        stream = os.popen(stardog_query)
        output = stream.read().strip()
        # print('results:\n', output)
        output = output.split('\n')
        # title
        # print('title line0', output[0])
        # print('title line1', output[1])
        # print('title line2', output[2])

        # get most relevant recipe
        print(len(output))
        # skip the first one, identical
        for rank in range(4, len(output) - 3):
            sim_recipe = output[rank].split('|')
            pos_title = sim_recipe[1].strip()[1:-1]
            pos_score = float(sim_recipe[2].strip())
            pos_assertion = sim_recipe[3].strip()
            print(pos_title, pos_score, pos_assertion)

            find_pos = get_sampleID(pos_assertion, anchor_url)
            if find_pos and find_pos != anchor_url:
                break

        # get positive recipe
        for rank in range(len(output) - 3 - 1, 2, -1):
            sim_recipe = output[rank].split('|')
            neg_title = sim_recipe[1].strip()[1:-1]
            neg_score = float(sim_recipe[2].strip())
            neg_assertion = sim_recipe[3].strip()
            print(neg_title, neg_score, neg_assertion)

            find_neg = get_sampleID(neg_assertion, anchor_url)
            if find_neg:
                break

        url_ls.remove(anchor_url)
        url_ls.remove(find_pos)
        if find_neg in url_ls:
            url_ls.remove(find_neg)
            neg_line = foodcom[find_neg]['id'] + '\t' + foodcom[find_neg]['title'] + '\t' + \
                       foodcom[find_neg]['partition'] + '\t' + find_neg + '\t' + str(neg_score) + '\n'
        else:
            find_neg = random.choice(url_ls)
            neg_line = foodcom[find_neg]['id'] + '\t' + foodcom[find_neg]['title'] + '\t' + \
                       foodcom[find_neg]['partition'] + '\t' + find_neg + '\t' + str(0) + '\n'

        anchor_line = foodcom[anchor_url]['id'] + '\t' + foodcom[anchor_url]['title'] + '\t' + \
                      foodcom[anchor_url]['partition'] + '\t' + anchor_url + '\t' + '1.0' + '\n'
        pos_line = foodcom[find_pos]['id'] + '\t' + foodcom[find_pos]['title'] + '\t' + \
                      foodcom[find_pos]['partition'] + '\t' + find_pos + '\t' + str(pos_score) + '\n'

        with open('triplet.txt', 'a+') as out_file:
            out_file.write(anchor_line)
            out_file.write(pos_line)
            out_file.write(neg_line)

        iteration += 1

        print('iteration', iteration)
        if iteration % 100 == 0:
            with open('remain_url.json', 'w') as out_file:
                json.dump(url_ls, out_file)
                print('save new remain url', len(url_ls))

        print(len(url_ls))
        # exit()
    return 0

prepare_sparql_query()