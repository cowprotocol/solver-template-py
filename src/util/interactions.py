def out_interaction(recipient, token, amount):
  # transfer(market_maker_eoa, amount)
  calldata = "a9059cbb000000000000000000000000{0}{1:0{2}x}".format(recipient[2:],int(amount),64)
  return {
    "target": str(token),
    "value": "0",
    "call_data":  encode_calldata(calldata)
  }

def in_interaction(allowance_proxy_address, token, amount):
  # send(token, amount)
  calldata = "d0679d34000000000000000000000000{0}{1:0{2}x}".format(str(token)[2:],int(amount),64)
  return {
    "target": allowance_proxy_address,
    "value": "0",
    "call_data":  encode_calldata(calldata)
  }

def encode_calldata(calldata):
  return [int(calldata[2*i:2*i+2], 16) for i in range(int(len(calldata)/2))]
