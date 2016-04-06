local Recursor = nn.Recursor

function Recursor:reinforce(reward)
  assert(self.step - 1 == reward:size(2))
  reward = torch.squeeze(reward:transpose(1,2))
  reward = torch.CudaTensor(reward:size()):copy(reward)
  if self.modules then
    for step=1,self.step-1 do
      stepModule = self:getStepModule(step)
      local res = reward[step]
      res = (torch.type(res) == 'torch.CudaTensor') and res or torch.CudaTensor({res})
      stepModule:reinforce(res)
    end
  end
end
