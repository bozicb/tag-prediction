require 'daru'
require 'daru/io/importers/json'
require 'stopwords'

def word_frequency data
    word_freq={}
    unique_tags=data.tags.to_a.flatten.uniq
    filter = Stopwords::Snowball::Filter.new "en"
    # 1. Generate a word frequency hash for every tag
    unique_tags.each do |tag|
        rows=[]
        data.each(:row) do |row|
             rows << row['content'] if row['tags'].include? tag 
        end
        frequencies_for_tag=Hash.new 0
        words=filter.filter rows.flatten.join(" ").downcase.gsub(/[^a-z ]/i,'').split
        words.each do |word|
           frequencies_for_tag[word]+=1 
        end
        frequencies_for_tag.each do |key,value|
           frequencies_for_tag[key]=value/words.length.to_f
        end
        word_freq[tag]=frequencies_for_tag
    end
    # 2. Select predictions based on frequency
    predictions=Array.new 0
    data.content.each do |text|
        matches=Hash.new 0
        unique_tags.each do |tag|
          filter.filter(text.downcase.gsub(/[^a-z ]/i,'').split).each do |word|
             matches[tag]+=word_freq[tag][word] if word_freq[tag].keys().include? word
            end
        end
        highest_matches=matches.select{|k,v| v>0.12}.keys[0..2]
        highest_matches=[matches.max_by{|k,v| v}[0]] if highest_matches.length==0
        predictions << highest_matches
    end
    data["predicted"]=Daru::Vector.new(predictions)
    data
end
