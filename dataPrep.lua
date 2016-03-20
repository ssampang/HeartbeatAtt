require 'dp'

records = {100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,118,119,121,122} --,117,123,124 --,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234}

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

wavePrefix = '/home/sid/Projects/HeartbeatNN/preprocessing/waveforms/'
AnnPrefix = '/home/sid/Projects/HeartbeatNN/preprocessing/annotations/'

waves = {}
anns = {}

for i=1,#records do
  waveFile = io.open(wavePrefix..tostring(records[i])..'.csv','r')
  annFile = io.open(AnnPrefix..tostring(records[i])..'.txt','r')
  
  wave = waveFile:read('*a')
  ann = annFile:read('*a')

  wave = split(wave,'\n')
  ann = split(ann,'\n')

  -- we limit the number of heartbeats to 1796 per record because it's convenient to have records of uniform length for training
  -- and empirically, disregarding the 3 records with <1796 beats and limiting all other records to 1796 beats yields the most number of beats

  if #ann >= 1796 then
    waveTensor = torch.Tensor(#wave-2,2):fill(0)
    annTensor = torch.Tensor(1796,1)

    for j=1,1796 do
      local val = removeSpaces(split(ann[j],' '))[4] == 'N' and 1 or 2
      annTensor[j] = val
    end
      anns[ records[i] ] = annTensor

    if ann[1797] then
      lastTime = tonumber(removeSpaces(split(ann[1797],' '))[1])
    end

    for j=3,#wave do
      local temp = split(wave[j],',')
      if lastTime and (tonumber(temp[1])>=lastTime) then
        --waveTensor = waveTensor:narrow(1,1,j-2)
        break
      end
      local val = {unpack(temp,2,3)}
      if #val ~=2 then
        print(tens)
      end
      val[1] = tonumber(val[1])
      val[2] = tonumber(val[2])
      tens = torch.Tensor(val)

      waveTensor[j-2] = torch.Tensor(val)
    end
    waves[ records[i] ] = waveTensor
  end
end

size = #records
shuffle = torch.randperm(size)
input = torch.FloatTensor(size,1,650000,2)
target = torch.IntTensor(size,1796)

validRatio = 0.2

for i=1,size do
  local idx = shuffle[i]
  input[idx]:copy( waves[ records[idx] ] )
  target[idx]:copy( anns[ records[idx] ] )
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

torch.save('HalfData',ds)

