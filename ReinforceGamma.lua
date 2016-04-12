------------------------------------------------------------------------
--[[ ReinforceGamma ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are mean (mu) of multivariate Gamma distribution. 
-- Ouputs are samples drawn from these distributions.
-- Standard deviation is provided as constructor argument.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.237-239) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
require('randomkit')
require('cephes')

local ReinforceGamma, parent = torch.class("nn.ReinforceGamma", "nn.Reinforce")

function ReinforceGamma:__init(stdev,minVal,maxVal, stochastic)
   parent.__init(self, stochastic)
   self.stdev = stdev
   self.minVal = minVal
   self.maxVal = maxVal
   if not stdev then
      self.gradInput = {torch.Tensor(), torch.Tensor()}
   end
end

function ReinforceGamma:updateOutput(input)
   local mean, stdev = input, self.stdev
   mean:mul(self.maxVal-self.minVal)
   mean:add(self.minVal)
   if torch.type(input) == 'table' then
      -- input is {mean, stdev}
      assert(#input == 2)
      mean, stdev = unpack(input)
   end
   assert(stdev)
   
   self.output:resizeAs(mean)
   
   if self.stochastic or self.train ~= false then
      local variance = mean.new()

      if torch.type(stdev) == 'number' then
        variance:pow(variance:resizeAs(mean):fill(stdev),2)
      elseif torch.isTensor(stdev) then
         if stdev:dim() == mean:dim() then
            assert(stdev:isSameSizeAs(mean))
            variance:pow(stdev,2)
         else
            assert(stdev:dim()+1 == mean:dim())
            self._stdev = self._stdev or stdev.new()
            self._stdev:view(stdev,1,table.unpack(stdev:size():totable()))
            self.__stdev = self.__stdev or stdev.new()
            self.__stdev:expandAs(self._stdev, mean)
            variance:pow(self.__stdev,2)
         end
      else
         error"unsupported mean type"
      end
      local shape = torch.cdiv(torch.pow(mean,2), variance)
      local scale = torch.cdiv(variance,mean)
      self.output:copy(randomkit.gamma(shape:float(),scale:float()))
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(mean)
   end

   return self.output
end

function ReinforceGamma:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : Gamma probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (mean)
   -- s : standard deviation (sigma) (stdev)
   -- k : shape parameter of gamma dist
   -- theta: scale parameter of gamma dist
   
   local mean, stdev = input, self.stdev
   local gradMean, gradStdev = self.gradInput, nil
   if torch.type(input) == 'table' then
      mean, stdev = unpack(input)
      gradMean, gradStdev = unpack(self.gradInput)
   end
   assert(stdev)   
    
   -- Derivative of log gamma w.r.t. mean :
   -- d ln(f(x,u,s))   d ln(f(x,k,theta))     d k                                      2 * u
   -- -------------- = ------------------ * ------- = digamma(k) - ln(theta) + ln(x) * -----
   --      d u                 d k            d u                                       s^2
   
   gradMean:resizeAs(mean)

   local variance = mean.new()
   if torch.type(stdev) == 'number' then
      variance:pow(variance:resizeAs(mean):fill(stdev),2)
   else
      if stdev:dim() == mean:dim() then
         variance:pow(stdev,2)
      else
         variance:pow(self.__stdev,2)
      end
   end
   local shape = torch.cdiv(torch.pow(mean,2), variance)
   local scale = torch.cdiv(variance,mean)

   local dkdu = torch.cdiv(torch.mul(mean,2),variance)
   local digammaOutput = variance.new():resizeAs(shape):copy(cephes.digamma(shape:float()))
   local dRdk = torch.add( torch.add( digammaOutput,-1, torch.log(scale)), torch.log(self.output))
   gradMean:cmul(dRdk,dkdu)
   gradMean:mul(self.maxVal-self.minVal)
   -- multiply by variance reduced reward
   gradMean:cmul(self:rewardAs(mean) )
   -- multiply by -1 ( gradient descent on mean )
   gradMean:mul(-1)
   
   -- Derivative of log Gamma w.r.t. stdev :
   -- d ln(f(x,u,s))   (x - u)^2 - s^2
   -- -------------- = ---------------
   --      d s              s^3
   
   if gradStdev then
      gradStdev:resizeAs(stdev)
      -- (x - u)^2
      gradStdev:copy(self.output):add(-1, mean):pow(2)
      -- subtract s^2
      self._stdev2 = self._stdev2 or stdev.new()
      self._stdev2:resizeAs(stdev):copy(stdev):cmul(stdev)
      gradStdev:add(-1, self._stdev2)
      -- divide by s^3
      self._stdev2:cmul(stdev):add(0.00000001)
      gradStdev:cdiv(self._stdev2)
      -- multiply by reward
      gradStdev:cmul(self:rewardAs(stdev))
       -- multiply by -1 ( gradient descent on stdev )
      gradStdev:mul(-1)
   end
   
   return self.gradInput
end
