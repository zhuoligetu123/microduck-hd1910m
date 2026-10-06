#!/usr/bin/env python3
"""Exercise the running Rust server and real CPU MuJoCo policy (no motor I/O)."""
import asyncio
import json
import time
import urllib.request
import websockets


async def until(ws, predicate, seconds=4):
    async with asyncio.timeout(seconds):
        while True:
            value=json.loads(await ws.recv())
            if predicate(value):
                return value


async def main():
    async with websockets.connect('ws://127.0.0.1:38880/ws') as owner:
        state=await until(owner,lambda s:s.get('type')=='state' and s.get('online'))
        assert state['source']=='sim' and len(state['joints'])==14
        assert state['control_hz']>45, state['control_hz']
        assert len(state['policy_sha256'])==64
        await owner.send(json.dumps(dict(id=100,type='claim_control')))
        assert (await until(owner,lambda s:s.get('id')==100))['accepted']
        # A previous joystick lease must not cancel a newly accepted skill.
        await owner.send(json.dumps(dict(type='move',seq=1,vx=.1,vy=0,vyaw=0)))
        await until(owner,lambda s:s.get('mode')=='walk')
        await owner.send(json.dumps(dict(id=120,type='skill',name='walk')))
        assert (await until(owner,lambda s:s.get('id')==120))['accepted']
        finish=time.monotonic()+.7
        while time.monotonic()<finish:
            frame=await until(owner,lambda s:s.get('type')=='state')
            assert frame['mode']=='walk',frame['mode']
        for i,mode in enumerate(('walk','stand')):
            await owner.send(json.dumps(dict(id=101+i,type='skill',name=mode)))
            assert (await until(owner,lambda s:s.get('id')==101+i))['accepted']
            await until(owner,lambda s:s.get('mode')==mode)
        await owner.send(json.dumps(dict(id=1,type='mouth',open=1)))
        assert (await until(owner,lambda s:s.get('id')==1))['accepted']
        await until(owner,lambda s:s.get('type')=='state' and s.get('mouth')==1)
        async with websockets.connect('ws://127.0.0.1:38880/ws') as viewer:
            await viewer.send(json.dumps(dict(id=2,type='mouth',open=0)))
            assert not (await until(viewer,lambda s:s.get('id')==2))['accepted']
            # A viewer disconnect must not clear the owner's in-flight motion.
            await owner.send(json.dumps(dict(type='move',seq=2,vx=.1,vy=0,vyaw=0)))
            await until(owner,lambda s:s.get('mode')=='walk')
        start=await until(owner,lambda s:s.get('type')=='state')
        assert start['mode']=='walk'
        end=time.monotonic()+2
        seq=3
        frames=[]
        while time.monotonic()<end:
            await owner.send(json.dumps(dict(type='move',seq=seq,vx=.1,vy=0,vyaw=0)))
            seq+=1
            frames.append(await until(owner,lambda s:s.get('type')=='state'))
            await asyncio.sleep(.02)
        stopped=await until(owner,lambda s:s.get('mode')=='stand',seconds=2)
        delta=sum((a-b)**2 for a,b in zip(frames[-1]['root'][:2],start['root'][:2]))**.5
        assert delta>.01,delta
        await owner.send(json.dumps(dict(id=3,type='skill',name='not_loaded')))
        assert not (await until(owner,lambda s:s.get('id')==3))['accepted']
        await owner.send(json.dumps(dict(id=4,type='mouth',open=0)))
        assert (await until(owner,lambda s:s.get('id')==4))['accepted']
        await owner.send(json.dumps(dict(id=5,type='move',seq=seq,vx=2,vy=0,vyaw=0)))
        assert not (await until(owner,lambda s:s.get('id')==5))['accepted']
        async with websockets.connect('ws://127.0.0.1:38880/ws') as replacement:
            await replacement.send(json.dumps(dict(id=110,type='claim_control')))
            assert (await until(replacement,lambda s:s.get('id')==110))['accepted']
            await owner.send(json.dumps(dict(id=111,type='move',seq=seq+1,vx=.1,vy=0,vyaw=0)))
            denied=await until(owner,lambda s:s.get('id')==111)
            assert not denied['accepted'] and denied['reason']=='another client owns control'
            await until(replacement,lambda s:s.get('mode')=='stand')
    devices=json.load(urllib.request.urlopen('http://127.0.0.1:38880/api/discover',timeout=4))
    assert any(d['protocol']=='microduck-app-v1' for d in devices)
    print(json.dumps(dict(passed=True,source='real_mujoco_onnx',policy_sha256=state['policy_sha256'],joints=14,frames=len(frames),
        displacement_m=delta,control_hz=stopped['control_hz'],checks=['mouth_ack','owner_lease',
        'observer_disconnect','joystick_motion','deadman','unknown_skill_rejected','range_rejected',
        'skill_walk_stand_ack','skill_cancels_stale_joystick_deadman','explicit_takeover_stops_previous','old_owner_rejected','policy_hash','discovery']),indent=2))


if __name__=='__main__':
    asyncio.run(main())
