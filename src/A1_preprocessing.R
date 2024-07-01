# file: A1_preprocessing.R 
# date: 25.11.2021
# updated: 03.12.2024 this file is the same as the 00_data_preparation.R that originated the data file BD4QoL_181223_encoded.csv
# updated: 24.11.2024 - we keep patients with missing status about recurrency at baseline
#                       we assume that a recurrent patient would have this recorded
#                       this is consistent with the BD4QoL data used in the cohort description paper
# author(s): Moreira-Soares, M & Fossen, EIF
# sensitive(?): No
# Description
# This script reads the Head & Neck 5000 (HN5000) data and filters the data
# based on the inclusion criteria defined in the BD4QoL study protocol.
# The second part of the code prepares the ordinal categorical variables, 
# replacing the categories by discrete levels
# At the end a csv file is saved with the filtered data.

library(dplyr)

# print r session info
sessionInfo()

# checking OS

if(Sys.info()[['sysname']]=="Windows"){
  path <- "M:/p1402-mauricim/analysis/data/"
} else{
  path  <- "/tsd/p1402/home/p1402-mauricim/analysis/data/"
  }

source(paste0(path,"../src/functions.R"))


# convert the data to csv
filename <- "HN057 - HN5000 V2.6 Dataset 020322.csv"
#write.csv(df, paste0(path, filename), row.names = FALSE, fileEncoding = "UTF-8")

df <- read.csv(paste0(path, filename), encoding="UTF-8")

time_zero <- 365.25 # considering 12 months as time 0 for survival

# alive at the baseline (12 months) and disease free
df <- df %>% filter((as.numeric(hn1_dv_death_cons)>=time_zero) &  
                        (hn3_nb16a_cb_recurrence != "1 - Yes") &  
                        (hn3_nb16a_cb_recurrence != "3 - NA") )
                      # updated 24.11.2023: removed condition
                      # hn3_nb16a_cb_recurrence != '.a - Missing'


# screening for cancer site
prospective <- TRUE
if(prospective){
  df <- df %>% filter((hn1_ICD_group_conf != '6 - Thyroid')&
                        (hn1_ICD_group_conf != '10 - Unknown Primary')&
                        (hn1_ICD_group_conf != '15 - Other') &
                        (hn1_ICD_group_conf != '12 - Minor and major salivary glands combined')
                        )
  
  # combine the categories nasal cavity+sinuses
  df <- df %>%
    mutate(hn1_ICD_group_conf = case_when(
      hn1_ICD_group_conf %in% c(
                                "8 - Nasal cavity",
                                "9 - Sinuses") ~ "6 - Nasal cavity/Sinuses",
      TRUE ~ hn1_ICD_group_conf
    ))
  
} else {
  # filter out also salivary glands category
  df <- df %>% filter((hn1_ICD_group_conf != '6 - Thyroid')&
                       (hn1_ICD_group_conf != '10 - Unknown Primary')&
                       (hn1_ICD_group_conf != '15 - Other')&
                       (hn1_ICD_group_conf != '8 - Nasal cavity')&
                       (hn1_ICD_group_conf != '9 - Sinuses')&
                       (hn1_ICD_group_conf != '12 - Minor and major salivary glands combined'))
}


# removing cancers without pathological information
df <- df %>% filter((hn1_SNOMED_group != "4 - Other")&
                      (hn1_SNOMED_group !=".b - Missing - under investigation"))

# only patients treated with curative intent will be selected
df <- df %>% filter(hn2_nb1_cb_plan_intent_1 == '1 - Curative')

# excluding presence of distant metastasis
df <- df %>% filter(hn1_nb5d_cb_m_best == "0") 

# create variable for shifted number of days alive by 12 months
df <- df %>% mutate(hn3_dv_death_time0 = as.numeric(hn1_dv_death_cons) - time_zero)


# create variables for life status at baseline and follow up
df <- df %>% mutate(hn3_dv_status = aliveAtTimepoint(0, hn3_dv_death_time0),
                    hn4_dv_status = aliveAtTimepoint(365.25*2, hn3_dv_death_time0))



# split treatment groups
df <- df %>% mutate(hn2_chemotherapy = if_else(grepl("chemo", tolower(hn2_treatgroup)), 1, 0))
df <- df %>% mutate(hn2_radiotherapy = if_else(grepl("radio", tolower(hn2_treatgroup)), 1, 0))
# merge three kinds of surgery in one variable: any surgery?
df <- df %>% mutate(hn2_surgery = if_else( grepl("primary", tolower(hn2_treatgroup)) | 
                                            grepl("neck", tolower(hn2_treatgroup)) | 
                                            grepl("surgery", tolower(hn2_treatgroup)) , 1, 0))

# We will encode the ordinal categorical variables with integers instead of categories

# Wealth index IMD10Quintil
df <- encodeAllLevels(df, "hn1_IMD10quint") 

# TNM stage
df <- encodeAllLevels(df, "hn1_TNM_stage_best") 

# Co-morbidity index
df <- df %>% mutate(hn1_nb4_cb_comorb_index = ifelse(hn1_nb4_cb_comorb_index=="5 - Unknown", 
                                                     NA_character_, 
                                                     hn1_nb4_cb_comorb_index))
df <- encodeAllLevels(df, "hn1_nb4_cb_comorb_index") 

# Household income
df <- encodeAllLevels(df, "hn1_dv_a21_ay_hhold_income")

# Education level
# UK Further Education (FE) college will be merged with Other college and School or college sixth form
# so the new ordinal encoding will refer to number of education years
# instead of the detailed education level

# Kathy merged primary and secondary school -> 1 - low
# All colleges -> 2 - medium
# uni or poly -> 3 - high
# we will keep the 4 levels because it has more information
df <- df %>% 
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "1 - Primary School", 1)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "2 - Secondary School", 2)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "3 - School or college sixth form", 3)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "4 - College FE", 3)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "5 - Uni or Poly", 4)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == "6 - Other college", 3)) %>%
  mutate(hn1_a7a_ay_education_level = replace(hn1_a7a_ay_education_level, hn1_a7a_ay_education_level == ".a - Missing", NA)) 

# how many days you have drinks per week in average - 3 time points  
df <- encodeAllLevels(df, "hn1_dv_drink_days")
df <- encodeAllLevels(df, "hn2_dv_drink_days")
df <- encodeAllLevels(df, "hn3_dv_drink_days")

df <- df %>% 
  mutate(hn1_dv_total_wk = na_if(hn1_dv_total_wk, ".a - Missing")) %>%
  mutate(hn2_dv_total_wk = na_if(hn2_dv_total_wk, ".a - Missing")) %>%
  mutate(hn3_dv_total_wk = na_if(hn3_dv_total_wk, ".a - Missing"))

df <- df %>% 
  mutate(hn1_dv_c30_ghs = na_if(hn1_dv_c30_ghs, ".a - Missing")) %>%
  mutate(hn2_dv_c30_ghs = na_if(hn2_dv_c30_ghs, ".a - Missing")) %>%
  mutate(hn3_dv_c30_ghs = na_if(hn3_dv_c30_ghs, ".a - Missing")) 


# for smoking
# conditionVariable  = hn1_a8_ay_tobacco
# conditionValue = "3 - Never"
# for drinking 
# conditionVariable =
# 

 
# sayh
# levels: .a - Missing    3 - Quite a bit 1 - Not at all  2 - A little    4 - Very much
sayh <- df %>% select(contains("sayh")) %>% colnames(.)
df <- encodeAllLevels(df, sayh)

# ygh
# levels: some variables contain different levels, but they are all ordinals
# ex: 1 - not at all ... 4 - very much
# .a - Missing  1 - very poor , 3, 4, 5, 6, 7 - excellent
ygh <- df %>% select(contains("ygh")) %>% colnames(.)
df <- encodeAllLevels(df, ygh)
df <- df %>% mutate_at(ygh, as.factor)


# yo - life orientation 
# Levels: .a - Missing 1 - Strongly disagree 2 - Disagree 3 - Neutral 4 - Agree 5 - Strongly agree
yo <- df %>% select(contains("_yo_")) %>% colnames(.)
df <- encodeAllLevels(df, yo)


# yf - your feelings - hospital anxiety and depression scale
# Levels:  
# Each variable presents different levels ordered from  high level of agreement/frequency to less agreement/frequency 
# from one to four. So we built the function checkLevel to encode then at once
# Some of the levels
#  "1 - Most of the time" "2 - A lot of the time"  "3 - From time to time occasionally" "4 - Not at all"
# "1 - Definitely as much" "2 - Not quite so much"  "3 - Only a little"      "4 - Hardly at all" 
#  "1 - Very definitely and quite badly"   "2 - Yes but not to badly"    "3 - A little but it does not worry me" "4 - Not at all"
# "1 - As much as I always could"  "2 - Not quite so much now"      "3 - Definitely not so much now" "4 - Not at all"

yf <- df %>% select(contains("_yf_")) %>% colnames(.)
df <- encodeAllLevels(df, yf)


# dyl - Difficulties in your life
# 1 - No difficulty, 2 - A little, 3 - Quite a bit, 4 - Very much 
dyl <- df %>% select(contains("_dyl_")) %>% colnames(.)
df <- encodeAllLevels(df, dyl)

# you and cancer - based on Fear of recurrence questionaire
# levels: 1 - Not at all, 2 - A little,  3 - Sometimes, 4 - A lot,  5 - All the time
yc <- df %>% select(contains("_yc_")) %>% colnames(.)
df <- encodeAllLevels(df, yc)

# about you - your health
# levels: 1 - I have no problems ... 2 - I have slight problems ... 3 - I have moderate problems ... 4 - I have severe problems ... 5 - I am unable to ...
ay_hlth <- df %>% select(contains("_ay_hlth_")) %>% colnames(.)
df <- encodeAllLevels(df, ay_hlth)

# yql - your quality of life
yql_ordinal <- c("hn1_i1_yql_pain", "hn2_i1_yql_pain", 
                  "hn1_i2_yql_appearance", "hn2_i2_yql_appearance", 
                  "hn1_i3_yql_activity", "hn2_i3_yql_activity",
                 "hn1_i4_yql_recreation", "hn2_i4_yql_recreation", 
                 "hn1_i5_yql_swallowing", "hn2_i5_yql_swallowing", 
                 "hn1_i6_yql_chewing", "hn2_i6_yql_chewing",
                 "hn1_i7_yql_speech", "hn2_i7_yql_speech", 
                 "hn1_i8_yql_shoulder", "hn2_i8_yql_shoulder", 
                 "hn1_i9_yql_taste", "hn2_i9_yql_taste",
                 "hn1_i10_yql_saliva", "hn2_i10_yql_saliva",  
                 "hn1_i11_yql_mood", "hn2_i11_yql_mood", 
                 "hn1_i12_yql_anxiety", "hn2_i12_yql_anxiety",
                "hn1_i14_yql_qol_month", "hn2_i14_yql_qol_month", 
                "hn1_i15_yql_qol_7_days", "hn2_i15_yql_qol_7_days",  
                "hn1_i16_yql_well_being", "hn2_i16_yql_well_being"
                )


yql_categorical <- c("hn1_i13a_yql_issues",  "hn1_i13b_yql_issues", "hn1_i13c_yql_issues",
                     "hn2_i13a_yql_issues",  "hn2_i13b_yql_issues", "hn2_i13c_yql_issues")

df <- encodeAllLevels(df, yql_ordinal)

# ya - your appearence
ya_ordinal <- c("hn1_k8b_ya_fem_masc_feel", "hn2_k8b_ya_fem_masc_feel",
"hn1_k20b_ya_normal_feel","hn2_k20b_ya_normal_feel",
"hn1_k22b_ya_avoid_going_out","hn2_k22b_ya_avoid_going_out",
"hn1_k24b_ya_avoid_pubs","hn2_k24b_ya_avoid_pubs",
"hn1_k1c_ya_features","hn2_k1c_ya_features",
"hn1_k2c_ya_physical_ability","hn2_k2c_ya_physical_ability",
"hn1_k3c_ya_disfigurement","hn2_k3c_ya_disfigurement",
"hn1_k5c_ya_treatment","hn2_k5c_ya_treatment",
"hn1_k6c_ya_bother_you","hn2_k6c_ya_bother_you")

df <- encodeAllLevels(df, ya_ordinal)

print("Harmonizing ethnicity")
# Harmonize hn1_na9_cb_ethnicity to have fewer categories

df<- df%>% mutate(hn1_na9_cb_ethnicity = recode(hn1_na9_cb_ethnicity, 
                                                "A1 - White –British"   ="1 - White",
                                                 "B1 - White -Irish"="1 - White",
                                                 "C1 - Any other White background"="1 - White",
                                                 "D1 - Mixed -White and Black Caribbean"="4 - Any other ethnic group",
                                                 "G1 - Any other Mixed background" ="4 - Any other ethnic group",
                                                 "S1 - Any other Ethnic group" ="4 - Any other ethnic group",
                                                 "H1 - Asian -Indian or British Indian"="3 - Asian",
                                                 "J1 - Asian -Pakistani or British Pakistani"="3 - Asian",
                                                 "L1 - Any other Asian background"="3 - Asian",
                                                 "R1 - Chinese" ="3 - Asian",
                                                 "M1 - Black -Caribbean or British Caribbean"="2 - Black",
                                                 "N1 - Black -African or British African" ="2 - Black",
                                                 "P1 - Any other Black background"="2 - Black",
                                                 "Z1 - Not stated/given" ="5 - Not stated/given/refused",
                                                 "Z2 - Patient Refused"="5 - Not stated/given/refused")) %>%
  mutate(hn1_na9_cb_ethnicity = na_if(hn1_na9_cb_ethnicity, ".a - Missing"))
print("finished harmonizing ethnicity")

# replacing the missing and unknown levels by NA
df <- df  %>% 
  mutate_all(~na_if(., ".a - Missing")) %>% 
  mutate_all(~na_if(.,"5 - Unknown"))  %>%
  mutate_all(~na_if(., "NA"))

# remove empty columns
df <- df %>% select_if(~!all(is.na(.))) 
df <- df %>% select_if(~!all(.==".a - Missing"))


# The easiest way to convert all the string columns into numerical is to write the data.frame to a csv file and import again

if(Sys.info()[['sysname']]=="Windows"){
  path <- "M:/p1402-mauricim/analysis/data/"
} else{
  path  <- "/tsd/p1402/home/p1402-mauricim/analysis/data/"
}

filename <- "BD4QoL_030124_encoded.csv"

predictors_file <- 'variables.csv'
predictors <- read.csv(paste0(path, predictors_file))
predictors[nrow(predictors)+1, ] <- c("hn2_chemotherapy", "categorical", "categorical", "factor")
predictors[nrow(predictors)+1, ] <- c("hn2_radiotherapy", "categorical", "categorical", "factor")
predictors[nrow(predictors)+1, ] <- c("hn2_surgery", "categorical", "categorical", "factor")

df <- df %>% select(predictors$variable)

missingness <- countMiss(df)

print("The following variables present more than 50% of missing values and will be removed:")
print(missingness %>% filter(missing > 50 & !grepl("hn4_dv_c30" ,variable) &  !grepl("hn4_dv_hn35" ,variable)))

remaining_variables <- merge(predictors, missingness, by="variable") %>% filter(missing < 50  | grepl("hn4_dv_c30" ,variable) |  grepl("hn4_dv_hn35" ,variable))

remaining_variables <- remaining_variables %>% arrange(., missing)

write.csv(remaining_variables, paste0(path, "variables_w_missing.csv"), row.names = FALSE)


df <- df %>% select(remaining_variables$variable) 
# equivocal category is very small (N=2), merging with not obtained
df <- df %>%
  mutate(hn1_nb9a_cb_hpv_status = ifelse(
    hn1_nb9a_cb_hpv_status %in% c("Equivocal", "Not obtained"),
    'Not obtained',
    hn1_nb9a_cb_hpv_status
  ))

# replace all NAs in HPV status by Not obtained
df <- df %>% mutate(hn1_nb9a_cb_hpv_status = tidyr::replace_na(hn1_nb9a_cb_hpv_status, 'Not obtained'))

# recode the sex variable to be binary with default level male = 0
df <- df %>%
  mutate(hn1_na8_cb_sex = recode(hn1_na8_cb_sex, "1 - Male" = 0, "2 - Female" = 1))


# reorder columns
df <- df %>% select(Studyid_hn057, hn1_na8_cb_sex, hn1_dv_age_cons, 
                    hn1_ICD_group_conf, hn1_TNM_stage_best,
                    hn2_chemotherapy, hn2_radiotherapy, hn2_surgery, everything())

# we will select only the variables that will be used for training
write.csv(df, paste0(path, filename), row.names = FALSE, fileEncoding = "UTF-8")

BD4QoL <- read.csv(paste0(path, filename), stringsAsFactors = TRUE)


# We will generate another data file with pure numerical variables, applying to all


# consistency test
# If new variables are re-enconded in the same way in the future, please included them here also
print("The next lines should show only TRUE outcomes. If not, you must debug.")
is.integer(BD4QoL$hn1_IMD10quint)
is.integer(BD4QoL$hn1_TNM_stage_best)
is.integer(BD4QoL$hn1_nb4_cb_comorb_index)
is.integer(BD4QoL$hn1_dv_a21_ay_hhold_income)
is.integer(BD4QoL$hn1_a7a_ay_education_level)
is.integer(BD4QoL$hn1_dv_drink_days)
is.integer(BD4QoL$hn2_dv_drink_days)
is.integer(BD4QoL$hn3_dv_drink_days)
is.numeric(BD4QoL$hn1_dv_total_wk)
is.numeric(BD4QoL$hn1_dv_c30_ghs)
is.numeric(BD4QoL$hn2_dv_c30_ghs)
is.numeric(BD4QoL$hn3_dv_c30_ghs)

