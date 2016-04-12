local Recursor = nn.Recursor

function Recursor:reinforce(reward)
  assert(self.step - 1 == reward:size(2))
  reward = torch.squeeze(reward:transpose(1,2)):contiguous()
  if self.modules then
    for step=1,self.step-1 do
      stepModule = self:getStepModule(step)
      local res = reward[step]
      res = torch.isTensor(res) and res or torch.Tensor{res}
      stepModule:reinforce(res)
    end
  end
end
