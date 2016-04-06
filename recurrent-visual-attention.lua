require 'dp'
require 'rnn'
require 'cutorch'
dofile 'SpatialGlimpse1D.lua'
dofile 'MultiVRReward.lua'
dofile 'Recursor.lua'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 20, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('--rewardScale', 0.25, "scale of positive reward (negative is 0)")
cmd:option('--unitPixels', 405, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', {2,180}, 'size of glimpse patch at highest res {height, width}')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 2, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
cmd:option('--rho', 4, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 64, 'number of hidden units used in Simple RNN.')
cmd:option('--FastLSTM', false, 'use LSTM instead of linear layer')

--[[ data ]]--
cmd:option('--dataset', 'MIT-BIH', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', false, 'dont propagate through the test set')
cmd:option('--overwrite', false, 'overwrite checkpoint')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

if opt.xpPath ~= '' then
   -- check that saved model exists
   assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

--[[data]]--
if opt.dataset == 'TranslatedMnist' then
   ds = torch.checkpoint(
      paths.concat(dp.DATA_DIR, 'checkpoint/dp.TranslatedMnist.t7'),
      function() return dp[opt.dataset]() end,
      opt.overwrite
   )
else
   ds = torch.load('Len'..opt.rho)
end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'optim'
      require 'cunn'
      cutorch.setDevice(opt.useDevice)
   end
   xp = torch.load(opt.xpPath)
   if opt.cuda then
      xp:cuda()
   else
      xp:float()
   end
   print"running"
   xp:run(ds)
   os.exit()
end

--[[Model]]--

glimpseAxes = 1
-- glimpse network (rnn input layer)
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(glimpseAxes, opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())

glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse1D(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(),true))
glimpseSensor:add(nn.Collapse(3))
glimpseSensor:add(nn.Linear(ds:imageSize('c')*(opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

-- rnn recurrent layer
if opt.FastLSTM then
  recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
else
  recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
end


-- recurrent neural network
rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), opt.rho)

-- actions (locator)
locator = nn.Sequential()
locator:add(nn.Linear(opt.hiddenSize, glimpseAxes))
locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(opt.unitPixels*2/ds:imageSize("w")))

attention = nn.RecurrentAttention(rnn, locator, opt.rho, {opt.hiddenSize})

-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
agent:add(attention)

-- classifier :
-- agent:add(nn.SelectTable(-1))

-- output should be a table of 1796 actions, and we want to apply a linear layer and a softmax to each one
classifier = nn.Sequential()
classifier:add(nn.Linear(opt.hiddenSize, #ds:classes()))
classifier:add(nn.LogSoftMax())

-- we don't want to train 1796 linear layers, so have them all share weights

--multipleActions = nn.ParallelTable()
--for i=1,opt.rho do
--  multipleActions:add( classifier:clone('weight','bias','gradWeight','gradBias') )
--end
--agent:add( multipleActions )
agent:add( nn.Sequencer( classifier ) )
agent:add( nn.JoinTable(2,2) )
agent:add( nn.View(-1,opt.rho,#ds:classes()) )

-- add the baseline reward predictor
seq = nn.Sequential()
filler = torch.CudaTensor(opt.rho,1)
for i=1,opt.rho do filler[i] = 1 end

seq:add(nn.Constant(filler,2))

bias = nn.Sequential()
bias:add(nn.SplitTable(1,2))

temp = nn.ParallelTable()
for i=1,opt.rho do
  temp:add(nn.Add(1))
end
bias:add(temp)
bias:add(nn.JoinTable(2,2))
bias:add(nn.View(-1,opt.rho,1))
seq:add(bias)
--seq:add(nn.Add(4))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
agent:add(concat2)

if opt.uniform > 0 then
   for k,param in ipairs(agent:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--[[Propagators]]--
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

-- not sure how to backprop multiple actions to the same network...but this is my best guess.

--backprop = nn.ParallelCriterion()
--for i=1,opt.rho do
--  backprop:add( nn.ClassNLLCriterion())
--end

train = dp.Optimizer{
-- split up each action into elements of a table, and split up each target into elements of a table, and apply ClassNLLCriterion to each
-- pair of elements

   loss = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.SequencerCriterion(nn.ClassNLLCriterion()), nn.SplitTable(1,2), nn.SplitTable(1,1))) -- BACKPROP
      :add(nn.ModuleCriterion(nn.MultiVRReward(agent, opt.rewardScale), nil, nn.Convert())) -- REINFORCE
   ,
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report)
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         if opt.lastEpoch < report.epoch and not opt.silent then
            print("mean gradParam norm", opt.meanNorm)
         end
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams
   end,
   feedback = nil,
   sampler = dp.ShuffleSampler{
      epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
   },
   progress = opt.progress
}


valid = dp.Evaluator{
   feedback = nil,
   sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
   progress = opt.progress
}
if not opt.noTest then
   tester = dp.Evaluator{
      feedback = nil,
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }
end

--[[Experiment]]--
xp = dp.Experiment{
   model = agent,
   optimizer = train,
   validator = valid,
   tester = tester,
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Agent :"
   print(agent)
end

xp.opt = opt

xp:run(ds)
