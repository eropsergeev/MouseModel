import torch
import numpy as np
import random
from itertools import product
from copy import deepcopy

class Shape:
    def __init__(self):
        pass
    def get_distance(self, x, y):
        raise NotImplementedError("Abstract class have not this method!")

class Circle(Shape):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def get_score(self, x, y):
        d = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        if d > self.r:
            return 0
        else:
            return 1 - d / self.r
    def get_distance(self, x, y):
        return np.abs(np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) - self.r)
    def get_vector_ditance(self, x, y, dx, dy):
        x -= self.x
        y -= self.y
        l = np.sqrt(dx ** 2 + dy ** 2)
        dx /= l
        dy /= l
        a = dx ** 2 + dy ** 2
        b = 2 * dx * x + 2 * dy * y
        c = x ** 2 + y ** 2 - self.r ** 2
        d = b ** 2 - 4 * a * c
        if d < 0:
            return np.inf
        else:
            t1 = (-b - np.sqrt(d)) / (2 * a)
            t2 = (-b + np.sqrt(d)) / (2 * a)
            if t2 < t1:
                t2, t1 = t1, t2
            if t2 < 0:
                return np.inf
            else:
                if t1 < 0:
                    return t2
                else:
                    return t1


class Line(Shape):
    def __init__(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        self.a = y1 - y2
        self.b = x2 - x1
        self.c = -(self.a * x1 + self.b * y1)
    def get_distance(self, x, y):
        return abs((self.a * x + self.b * y + self.c) / np.sqrt(self.a ** 2 + self.b ** 2))

class Segment(Shape):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def get_distance(self, x, y):
        pr = (x - self.p1[0]) * (self.p2[0] - self.p1[0]) + (y - self.p1[1]) * (self.p2[1] - self.p1[1])
        if pr > 0 and pr < (self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[1]) ** 2:
            return Line(self.p1, self.p2).get_distance(x, y)
        return min(np.hypot(x - self.p1[0], y - self.p1[1]), np.hypot(x - self.p2[0], y - self.p2[1]))


class AbstractField:
    def __init__(self):
        pass
    
    def get_max_dist(self):
        raise NotImplementedError("Abstract class have not this method!")
    
    def move(self, x, y, dx, dy):
        raise NotImplementedError("Abstract class have not this method!")


class CircleField:
    def __init__(self, r, obst, exit = None):
        self.r = r
        self.obst = obst
        self.exit = exit
        
    def get_max_dist(self):
        return self.r * 2
        
    def move(self, x, y, dx, dy):
        l = np.sqrt(dx ** 2 + dy ** 2)
        if len(self.obst):
            min_dist = min(map(lambda c: c.get_vector_ditance(x, y, dx, dy), self.obst)) - 1e-5
        else:
            min_dist = self.r * 3
        min_dist = min(min_dist, Circle(0, 0, self.r).get_vector_ditance(x, y, dx, dy))
        if min_dist < l:
            dx *= min_dist / l
            dy *= min_dist / l
        nx = x + dx
        ny = y + dy
        l = np.hypot(nx, ny)
        if l > self.r:
            nx /= l / self.r
            ny /= l / self.r
        if self.exit:
            return (nx, ny), self.exit.get_score(x + dx, y + dy)
        else:
            return (nx, ny)


def array_draw_objects(x0, x1, y0, y1, sz, objs, r, g, b, progress_bar = False):
    d = 5e-3 * (x1 - x0)
    ans = np.zeros((sz, sz, 3), dtype=np.float)
    coords = product(range(sz), range(sz))
    if progress_bar:
        coords = tqdm_notebook(list(coords))
    for i, j in coords:
            x = x0 + i / sz * (x1 - x0)
            y = y0 + j / sz * (y1 - y0)
            for o in objs:
                s = o.get_distance(x, y)
                if s < d:
                    ans[i][j][0] = r
                    ans[i][j][1] = g
                    ans[i][j][2] = b
    return ans

class AbstractModel:
    def __init__(self, field : AbstractField, x, y):
        self.field = field
        self.x = x
        self.y = y
    def move(self):
        raise NotImplementedError("Abstract class have not this method!")
    def get_neurons(self):
        raise NotImplementedError("Abstract class have not this method!")


class RandomModel(AbstractModel):
    def __init__(self, field : AbstractField, x, y, step, neurons_count, cover = None):
        AbstractModel.__init__(self, field, x, y)
        self.step = step
        if cover == None:
            cnt = 0
            cover = []
            while cnt < 100:
                cnt += 1
                rr = random.random() * field.get_max_dist()
                ang = random.random() * 2 * np.pi
                x = np.cos(ang) * rr
                y = np.sin(ang) * rr
                f = 1
                for c in cover:
                    if c.get_score(x, y) > 0:
                        f = 0
                        break
                if f:
                    cnt = 0
                    cover.append(Circle(x, y, random.random() * 10 + 20))
        self.cover = cover
        self.neurons_count = neurons_count
    def move(self):
        ang = np.random.random() * 2 * np.pi
        dx = self.step * np.cos(ang)
        dy = self.step * np.sin(ang)
        self.x, self.y = self.field.move(self.x, self.y, dx, dy)
    def _add_random_neurons(self, positions, place_neurons):
        ans = np.random.rand(self.neurons_count)
        for i, x in enumerate(positions):
            ans[x] = place_neurons[i]
        return ans
    def get_neurons(self):
        positions = random.sample(list(range(self.neurons_count)), len(self.cover))
        return self._add_random_neurons(positions, [c.get_score(self.x, self.y) for c in self.cover])

def default_state_func(field, x, y):
    ans = np.zeros(16)
    md = field.get_max_dist()
    for i, (dx, dy) in enumerate(delta):
        ans[i] = field.exit.get_vector_ditance(x, y, dx, dy) or md
        if ans[i] > md:
            ans[i] = md
        (tx, ty), _ = field.move(x, y, dx * md, dy * md)
        ans[i + 8] = np.hypot(x - tx, y - ty)
    return torch.Tensor(ans)

def default_move_func(net, state):
    q = net(state).view(-1)
    delta = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return delta[torch.argmax(q)]

def default_reward_func(field, ox, oy, x, y):
    ex = field.exit.x
    ey = field.exit.y
    return np.hypot(ox - ex, oy - ey) - np.hypot(x - ex, y - ey)

def default_empty_place_generator(field):
    x = y = None
    while x == None:
        ang = np.random.random() * 2 * np.pi
        d = np.random.random() * r
        x = d * np.cos(ang)
        y = d * np.sin(ang)
        if field.exit.get_score(x, y):
            x = y = None
        else:
            for o in field.obst:
                if o.get_score(x, y):
                    x = y = None
                    break
    return x, y

class NNModel(AbstractModel):
    def __init__(
        self,
        field : AbstractField,
        x, y,
        net,
        state_func = default_state_func,
        move_func = default_move_func,
        reward_func = default_reward_func,
        empty_place_generator = default_empty_place_generator,
        action_pool_size = 1000,
        iters_count = 3000,
        batch_size = 100):
        AbstractModel.__init__(self, field, x, y)
        self.x = x
        self.y = y
        self.state_func = state_func
        self.move_func = move_func
        self.net = net
        action_pool = []
        max_action_poll_size = 10 ** 3
        cur_pos = 0
        def add_action(state, act, new_state, reg):
            global cur_pos
            global action_pool
            if len(action_pool) < max_action_poll_size:
                action_pool.append((state, act, new_state, reg))
            else:
                action_pool[cur_pos] = (state, act, new_state, reg)
                cur_pos += 1
                cur_pos %= max_action_poll_size
        gamma = 0.9
        opt = torch.optim.Adam(net.parameters())
        x, y = empty_place_generator(field)
        for i in range(iters_count):
            state = state_func(field, x, y)
            dx, dy = move_func(net, state)
            old_state = torch.tensor(state)
            ox, oy = x, y
            if field.exit != None:
                (x, y), e = field.move(x, y, dx * step, dy * step)
            else:
                x, y = field.move(x, y, dx * step, dy * step)
                e = False
            rew = reward_func(field, ox, oy, x, y)
            add_action(old_state, act, state, rew)
            if len(action_pool) == max_action_poll_size:
                batch = random.sample(action_pool, batch_size)
                new_states = list(map(lambda x: x[2].view(-1), batch))
                new_states = torch.cat(new_states)
                old_states = list(map(lambda x: x[0].view(-1), batch))
                old_states = torch.cat(old_states)
                rews = list(map(lambda x: x[3], batch))
                rews = torch.Tensor(rews)
                acts = list(map(lambda x: x[1], batch))
                acts = torch.LongTensor(acts)
                qs = net(new_states)
                qs = torch.max(qs, dim=1)[0]
                qs = qs * gamma + rews
                opt.zero_grad()
                rqs = net(old_states)
                rqs = torch.cat(list(map(lambda x: x[0][x[1]].view(1), zip(rqs, acts))))
                loss = ((qs - rqs) ** 2).mean()
                loss.backward()
                opt.step()
            if e:
                x, y = empty_place_generator(field)
        
    def move(self):
        state = self.state_func(self.field, self.x, self.y)
        dx, dy = self.move_func(self.net, state)
        if self.field.exit != None:
            (nx, ny), e = self.field.move(self.x, self.y, dx, dy)
        else:
            (nx, ny) = self.field.move(self.x, self.y, dx, dy)
            e = None
        self.x = nx
        self.y = ny
        return e
    def get_neurons(self):
        state = self.state_func(self.field, self.x, self.y)
        return self.net(state, return_neurons = True)[1]

class DefaultNetwork(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.l1 = torch.nn.Linear(16, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 8)
    
    def forward(self, x, return_neurons = False):
        x = x.view(-1, 16)
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        if return_neurons:
            neurons = x
        x = self.l2(x)
        x = torch.nn.functional.relu(x)
        if return_neurons:
            neurons = torch.cat((neurons, x), dim=1)
        x = self.l3(x)
        if x.size()[0] == 1:
            x = x.view(8)
        if return_neurons:
            return x, neurons
        else:
            return x



