local Recursor = nn.Recursor

function Recursor:reinforce(reward)
  assert(self.step - 1 == reward:size(1))
  if self.modules then
    for step=1,self.step-1 do
      stepModule = self:getStepModule(step)
      stepModule:reinforce(torch.CudaTensor({reward[step]}))
    end
  end
end
