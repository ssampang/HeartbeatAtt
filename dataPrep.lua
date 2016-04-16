require 'dp'
require 'cutorch'

records = {100,101,103,104,105,106,107,108,109,111,112,113,114,115,116,118,119,121,122,102,117,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234}
--records = {203,200,114,201,115,215,108,209,208,210,221,223,104,121,217,105,112,219,106,123,228,101,116,234,112,100,117,230,103,222,220,102,119,213,205,233,202,113}
labels = {['A']=true, ['E']=true, ['J']=true, ['L']=true, ['N']=true, ['R']=true, ['a']=true, ['e']=true, ['j']=true, ['S']=true}
wavePrefix = '/home/sid/Projects/HeartbeatNN/preprocessing/waveforms/'
AnnPrefix = '/home/sid/Projects/HeartbeatNN/preprocessing/annotations/'
validRatio = 0.2
timeLength = 4.5
numBeats = 4
length = math.floor(360 * timeLength /2) -- we subsample by 50%, and 360 is sampling frequency in Hz and max length of 4 heartbeats is 3.5335 seconds obtained from getMaxLength

function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

function removeSpaces(tb)
  local i = 1
  while tb[i] do
    if tb[i]=='' then
      table.remove(tb,i)
    else
      i = i + 1
    end
  end
  return tb
end

function waveTime(sample)
  return tonumber(split(sample,',')[1])
end

function waveSamples(sample)
  local splitted = split(sample,',')
  return {tonumber(splitted[2]), tonumber(splitted[3])}
end

function annTime(sample)
  return tonumber(removeSpaces(split(sample,' '))[1])
end

function annLabel(sample)
  return removeSpaces(split(sample,' '))[4]
end

function getLabelCounts()
  local res={}
  for i=1,#records do
    local ann = anns[i]
    for j=5,#ann-4,4 do
      if labels[annLabel(ann[j])] and labels[annLabel(ann[j+1])] and labels[annLabel(ann[j+2])] and labels[annLabel(ann[j+3])] then
        res[records[i]] = res[records[i]] or {}
        for k=0,3 do
          res[records[i]][annLabel(ann[j+k])] = 1 + ( res[records[i]][annLabel(ann[j+k])] or 0)
        end
      end
    end
  end
  return res
end

function getMaxLength(anns, numBeats)
  local maxLength = 0
  for i=1,#records do
    local ann = anns[i]
    for j=1,#ann,numBeats do
      if ann[j+numBeats] then
        local nextTime = tonumber( removeSpaces( split(ann[j+numBeats],' '))[1])
        local endTime = tonumber( removeSpaces( split(ann[j+numBeats-1],' '))[1])
        endTime = (nextTime-endTime)/2 + endTime
        if j>1 then
          local prevTime = tonumber( removeSpaces( split(ann[j-1],' '))[1])
          local startTime = tonumber( removeSpaces( split(ann[j],' '))[1])
          startTime = (startTime-prevTime)/2 + prevTime
          if maxLength < (endTime - startTime) then
            maxLength = endTime - startTime
            print( tostring(maxLength)..' '..tostring(i)..' '..tostring(startTime))
          end
        end
      end
    end
  end
  return maxLength
end

waves = {}
anns = {}

totalBeats = 0
abnormalBeats = 0
beatDist = {}
for i=0,numBeats do
  beatDist[i]=0
end

badBeatDist = {}
for i=0,numBeats do
  badBeatDist[i]=0
end

tooLong = 0

for i=1,#records do
  local waveFile = io.open(wavePrefix..tostring(records[i])..'.csv','r')
  local annFile = io.open(AnnPrefix..tostring(records[i])..'.txt','r')
  local wave = waveFile:read('*a')
  local ann = annFile:read('*a')
  wave = split(wave,'\n')
  ann = split(ann,'\n')

  local waveIndex = 3

  for j=1,#ann do
    if not (annLabel(ann[j]) == 'N') then
      abnormalBeats = abnormalBeats + 1
    end
    totalBeats = totalBeats + 1
  end

  for j= numBeats+1,#ann-numBeats,numBeats do
    local startTime = (annTime(ann[j]) - annTime(ann[j-1]))/2+annTime(ann[j-1])
    local endTime = (annTime(ann[j+numBeats]) - annTime(ann[j+numBeats-1]))/2+annTime(ann[j+numBeats-1])

    if (endTime-startTime) <= timeLength then
      local output = {}
      for k=1,numBeats do
        output[#output+1] = annLabel(ann[j+k-1])
      end

      if labels[output[1]] and labels[output[2]] and labels[output[3]] and labels[output[4]] then
        temp = torch.Tensor(length,2):fill(0)
        tempIndex = 1

        while waveIndex>3 and waveTime(wave[waveIndex-2]) >= startTime do
          waveIndex = waveIndex - 2
        end

        while (waveIndex+2)<=#ann and waveTime(wave[waveIndex]) < startTime do
          waveIndex = waveIndex + 2
        end

        while waveIndex+1<=#ann and waveTime(wave[waveIndex+1]) < endTime do
          temp[tempIndex]:copy(torch.mean(torch.Tensor({waveSamples(wave[waveIndex]),waveSamples(wave[waveIndex+1])}),1))
          tempIndex = tempIndex+1
          waveIndex = waveIndex+2
        end
        
        local numAbnormalBeats = 0 
        
        for k=1,numBeats do
          output[k] = output[k]=='N' and 1 or 2
          if output[k] == 2 then numAbnormalBeats = numAbnormalBeats + 1 end
        end
        
        beatDist[numAbnormalBeats] = beatDist[numAbnormalBeats] + 1
        waves[#waves+1] = temp
        anns[#anns+1] = torch.Tensor(output)
      else
        local numAbnormalBeats = 0
        for k=1,numBeats do
          if not labels[output[k]] then numAbnormalBeats = numAbnormalBeats + 1 end
        end
        badBeatDist[numAbnormalBeats] = badBeatDist[numAbnormalBeats] + 1
      end
    else tooLong = tooLong + 4
    end
  end
end

size = #waves
shuffle = torch.randperm(size)
input = torch.CudaTensor(size,1,length,2):fill(0)
target = torch.CudaTensor(size,numBeats):fill(0)

for i=1,size do
  local idx = shuffle[i]
  input[idx]:copy( waves[ idx ] )
  target[idx]:copy( anns[ idx ] )
end

nValid = math.floor(size*validRatio)
nTest = nValid
nTrain = size - nTest - nValid

trainInput = dp.ImageView('bcwh', input:narrow(1,1,nTrain))
trainTarget = dp.ClassView('bt',target:narrow(1,1,nTrain))
validInput = dp.ImageView('bcwh', input:narrow(1,nTrain+1,nValid))
validTarget = dp.ClassView('bt', target:narrow(1,nTrain+1,nValid))
testInput = dp.ImageView('bcwh', input:narrow(1,nTrain+nValid+1,nTest))
testTarget = dp.ClassView('bt', target:narrow(1,nTrain+nValid+1,nTest))

trainTarget:setClasses({'Normal','Arrhythmia'})
validTarget:setClasses({'Normal','Arrhythmia'})
testTarget:setClasses({'Normal','Arrhythmia'})

train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}
test = dp.DataSet{inputs=testInput,targets=testTarget,which_set='test'}

ds = dp.DataSource{train_set=train,valid_set=valid,test_set=test}
ds:classes{'Normal','Arrhythmia'}
--print(input[1])
--print(ds:get('train','input')[2])


print('Sample data input')
print(ds:get('train','input')[1])
print('Sample data target')
print(ds:get('train','target')[1])
print('Created dataset of size '..tostring(size))
print('Total number of heartbeats: '..tostring(totalBeats)..' normal beats: '..tostring(totalBeats-abnormalBeats)..' arrhythmic beats: '..tostring(abnormalBeats))
print('Distribution of '..tostring(numBeats)..' beat sequences with respect to number of abnormal beats within each sequence: ')
print(beatDist)
sumBeats = 0
for i=0,numBeats do
  sumBeats = sumBeats + beatDist[i]
end
print('Total number of heartbeats in dataset: '..tostring(4*sumBeats))

print('Distribution of '..tostring(numBeats)..' beat sequences that contain irrelevant beats with respect to number of irrelevant beats within each sequence: ')
print(badBeatDist)
sumBeats = 0
for i=0,numBeats do
  sumBeats = sumBeats + badBeatDist[i]
end
print('Number of beats that were a part of sequences that exceeded '..tostring(timeLength)..' seconds: '..tostring(tooLong))
print('Total number of sequences that contained irrelevant beats: '..tostring(sumBeats))
print('Total number of heartbeats skipped for dataset: '..tostring(4*sumBeats + tooLong))

torch.save('Len'..numBeats,ds)
