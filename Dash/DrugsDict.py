DrugFamilies = ['TCA', 'MAOI', 'SSRI', 'SNRI', 'Benzodiazepines', 'AtypicalAntipsychotics', 'GABA', 'MixedAntidepressants']

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

# Unique categories for dropdown
drug_families = ['All'] + sorted(list(set(DrugsDict.values())))