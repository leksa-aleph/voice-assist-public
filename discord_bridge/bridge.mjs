// Мост Discord -> voice-assist.
//
// Бот заходит в голосовой канал вслед за своим человеком и получает звук
// каждого участника отдельной дорожкой. Поэтому имя говорящего известно
// точно, даже когда говорят хором. Опус декодируется здесь, наружу уходит
// PCM 48 кГц моно.
//
// stdout - кадры: [тип:1][длина:4 LE][данные]
//   тип 1: JSON-событие
//   тип 2: звук: [id пользователя: 8 LE][int16 LE PCM]
// stdin  - JSON-команды построчно.
// stderr - журнал.

import {
  Client, Events, GatewayIntentBits, MessageFlags, PermissionFlagsBits, SlashCommandBuilder,
} from 'discord.js';
import {
  EndBehaviorType, VoiceConnectionStatus, entersState, joinVoiceChannel,
} from '@discordjs/voice';
import OpusScript from 'opusscript';
import readline from 'node:readline';

const TOKEN = process.env.DISCORD_TOKEN || '';
let followId = process.env.FOLLOW_USER_ID || null;
let followExplicit = Boolean(followId);
let ownerIds = new Set();
let skipId = process.env.SKIP_USER_ID || null;
let current = null;            // { guildId, channelId, connection }
let joining = null;
const decoders = new Map();
const subs = new Map();

const log = (...a) => process.stderr.write('[bridge] ' + a.join(' ') + '\n');

function frame(type, payload) {
  const head = Buffer.alloc(5);
  head.writeUInt8(type, 0);
  head.writeUInt32LE(payload.length, 1);
  process.stdout.write(Buffer.concat([head, payload]));
}

function emit(obj) {
  frame(1, Buffer.from(JSON.stringify(obj), 'utf8'));
}

function emitAudio(userId, pcm) {
  const head = Buffer.alloc(8);
  head.writeBigUInt64LE(BigInt(userId), 0);
  frame(2, Buffer.concat([head, pcm]));
}

if (!TOKEN) {
  emit({ evt: 'status', state: 'error', detail: 'no-token' });
  process.exitCode = 2;
  process.stdin.destroy();
  setTimeout(() => process.exit(2), 200).unref();
}

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
});

const commands = [
  new SlashCommandBuilder()
    .setName('join')
    .setNameLocalizations({ ru: 'субтитры' })
    .setDescription('Join your voice channel and follow you')
    .setDescriptionLocalizations({ ru: 'Зайти в твой голосовой канал и ходить за тобой' }),
  new SlashCommandBuilder()
    .setName('leave')
    .setNameLocalizations({ ru: 'стоп' })
    .setDescription('Leave the voice channel')
    .setDescriptionLocalizations({ ru: 'Выйти из голосового канала' }),
].map((c) => c.toJSON());

async function registerCommands(guild) {
  try {
    await guild.commands.set(commands);
  } catch (e) {
    log('commands', guild.id, e.message);
  }
}

function memberInfo(m) {
  return {
    id: m.id,
    name: m.displayName || m.user?.globalName || m.user?.username || m.id,
    avatar: m.displayAvatarURL({ extension: 'png', size: 64 }),
    bot: Boolean(m.user?.bot),
  };
}

function emitMembers() {
  if (!current) return;
  const guild = client.guilds.cache.get(current.guildId);
  const ch = guild?.channels.cache.get(current.channelId);
  if (!ch) return;
  const members = [...ch.members.values()]
    .filter((m) => m.id !== client.user.id)
    .map(memberInfo);
  emit({ evt: 'members', members });
}

function findUserVoice(userId) {
  for (const guild of client.guilds.cache.values()) {
    const vs = guild.voiceStates.cache.get(userId);
    if (vs?.channelId) return { guild, channelId: vs.channelId };
  }
  return null;
}

function dropSubscriptions() {
  for (const s of subs.values()) {
    try { s.destroy(); } catch {}
  }
  subs.clear();
  for (const d of decoders.values()) {
    try { d.delete(); } catch {}
  }
  decoders.clear();
}

function leave(announce = true) {
  const c = current;
  current = null;
  dropSubscriptions();
  if (c) {
    try { c.connection.destroy(); } catch {}
  }
  if (announce) {
    emit({ evt: 'channel', channel: null });
    emit({ evt: 'status', state: 'online' });
  }
}

function subscribe(receiver, userId) {
  if (subs.has(userId) || userId === client.user.id) return;
  const stream = receiver.subscribe(userId, {
    end: { behavior: EndBehaviorType.AfterSilence, duration: 800 },
  });
  subs.set(userId, stream);
  let dec = decoders.get(userId);
  if (!dec) {
    dec = new OpusScript(48000, 1, OpusScript.Application.VOIP);
    decoders.set(userId, dec);
  }
  stream.on('data', (packet) => {
    if (userId === skipId) return;
    let pcm;
    try {
      pcm = dec.decode(packet);
    } catch {
      return;
    }
    if (pcm && pcm.length) emitAudio(userId, pcm);
  });
  const done = () => { if (subs.get(userId) === stream) subs.delete(userId); };
  stream.once('end', done);
  stream.once('close', done);
  stream.on('error', (e) => { log('stream', userId, e.message); done(); });
}

async function joinChannel(guild, channelId) {
  if (current && current.channelId === channelId) return;
  if (joining === channelId) return;
  joining = channelId;
  if (current) leave(false);

  const channel = guild.channels.cache.get(channelId);

  // Без права «Подключаться» Discord молча игнорирует просьбу войти, и
  // подключение висит на signalling. Проверяем заранее и говорим прямо.
  const me = guild.members.me;
  const perms = channel && me ? channel.permissionsFor(me) : null;
  if (perms && !perms.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.Connect])) {
    log('voice', 'no Connect permission in', channel.name);
    joining = null;
    emit({ evt: 'status', state: 'error', detail: 'no-connect' });
    emit({ evt: 'channel', channel: null });
    return;
  }
  if (channel?.full && !(perms && perms.has(PermissionFlagsBits.MoveMembers))) {
    log('voice', 'channel is full:', channel.name);
    joining = null;
    emit({ evt: 'status', state: 'error', detail: 'channel-full' });
    emit({ evt: 'channel', channel: null });
    return;
  }
  emit({ evt: 'status', state: 'joining', detail: channel?.name || '' });

  const connection = joinVoiceChannel({
    channelId,
    guildId: guild.id,
    adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,      // глухой бот не получает звук
    selfMute: true,
    daveEncryption: true,
    debug: true,
  });
  current = { guildId: guild.id, channelId, connection };

  // Подробный журнал подключения к голосу: по нему видно, где застряло.
  // signalling - Discord не дал данные голосового сервера (нет права
  // «Подключаться», канал полон); connecting - не доходит сеть до голосового
  // сервера (UDP/WebSocket режет блокировка или обход не пускает node.exe).
  let lastStatus = connection.state.status;
  let ready = false;
  let dbg = 0;
  log('voice', 'join', channel?.name || channelId);
  connection.on('stateChange', (o, n) => {
    lastStatus = n.status;
    if (o.status !== n.status) log('voice', o.status, '->', n.status);
  });
  connection.on('debug', (m) => {
    if (ready || dbg++ > 80) return;
    // токены голосовой сессии в журнал не пишем
    log('voice dbg', String(m).replace(/"(token|session_id)":"[^"]*"/g, '"$1":"***"').slice(0, 400));
  });

  connection.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      // переезд между каналами или короткий обрыв - соединение восстановится само
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, 5000),
        entersState(connection, VoiceConnectionStatus.Connecting, 5000),
      ]);
    } catch {
      if (current?.connection === connection) {
        leave(true);
        setTimeout(followTarget, 3000);
      } else {
        try { connection.destroy(); } catch {}
      }
    }
  });
  connection.on('error', (e) => log('voice', e.message));

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20000);
  } catch (e) {
    joining = null;
    const stuck = lastStatus;
    log('voice', 'timeout, stuck at', stuck);
    if (current?.connection === connection) {
      // причину считаем до leave(): destroy() сменит статус на destroyed
      const detail = stuck === VoiceConnectionStatus.Signalling ? 'voice-denied' : 'voice-net';
      leave(false);
      emit({ evt: 'status', state: 'error', detail });
      emit({ evt: 'channel', channel: null });
    }
    return;
  }
  ready = true;
  log('voice', 'ready');
  joining = null;
  if (current?.connection !== connection) return;

  const receiver = connection.receiver;
  receiver.speaking.on('start', (userId) => {
    if (userId === client.user.id) return;
    emit({ evt: 'speaking', user_id: userId, active: true });
    subscribe(receiver, userId);
  });
  receiver.speaking.on('end', (userId) => {
    if (userId === client.user.id) return;
    emit({ evt: 'speaking', user_id: userId, active: false });
  });

  emit({
    evt: 'channel',
    guild: { id: guild.id, name: guild.name },
    channel: { id: channelId, name: channel?.name || '' },
  });
  emit({ evt: 'status', state: 'in_voice', detail: channel?.name || '' });
  emitMembers();
}

async function followTarget() {
  if (!followId) return;
  const loc = findUserVoice(followId);
  if (loc) {
    await joinChannel(loc.guild, loc.channelId);
  } else if (current) {
    leave(true);
  }
}

client.once(Events.ClientReady, async (c) => {
  try {
    const app = await c.application.fetch();
    if (app.owner?.members) {
      for (const id of app.owner.members.keys()) ownerIds.add(id);
    } else if (app.owner?.id) {
      ownerIds.add(app.owner.id);
    }
  } catch (e) {
    log('application', e.message);
  }
  if (!followId && ownerIds.size === 1) {
    followId = [...ownerIds][0];
  }
  let followName = '';
  if (followId) {
    try {
      const u = await c.users.fetch(followId);
      followName = u.globalName || u.username || '';
    } catch {}
  }
  emit({
    evt: 'ready',
    bot: { id: c.user.id, name: c.user.username, avatar: c.user.displayAvatarURL({ extension: 'png', size: 128 }) },
    guilds: c.guilds.cache.size,
    follow_id: followId,
    follow_name: followName,
  });
  emit({ evt: 'status', state: 'online' });
  for (const g of c.guilds.cache.values()) registerCommands(g);
  followTarget();
});

client.on(Events.GuildCreate, (guild) => {
  registerCommands(guild);
  emit({ evt: 'guilds', count: client.guilds.cache.size });
});

client.on(Events.GuildDelete, () => {
  emit({ evt: 'guilds', count: client.guilds.cache.size });
});

client.on(Events.VoiceStateUpdate, (oldS, newS) => {
  const uid = newS.id;
  if (uid === client.user.id) {
    if (!newS.channelId && current && oldS.guild.id === current.guildId) {
      // бота выгнали из канала руками
      leave(true);
    }
    return;
  }
  if (uid === followId) {
    if (newS.channelId && newS.channelId !== current?.channelId) {
      joinChannel(newS.guild, newS.channelId);
    } else if (!newS.channelId && current && oldS.guild.id === current.guildId) {
      leave(true);
    }
  }
  if (current && (oldS.channelId === current.channelId || newS.channelId === current.channelId)) {
    emitMembers();
    if (oldS.channelId === current.channelId && newS.channelId !== current.channelId) {
      emit({ evt: 'speaking', user_id: uid, active: false });
      emit({ evt: 'left', user_id: uid });
    }
  }
});

client.on(Events.InteractionCreate, async (it) => {
  if (!it.isChatInputCommand()) return;
  const uid = it.user.id;
  const allowed = !followExplicit || uid === followId || ownerIds.has(uid);
  if (!allowed) {
    await it.reply({ content: 'Я хожу за другим человеком.', flags: MessageFlags.Ephemeral }).catch(() => {});
    return;
  }
  if (it.commandName === 'join') {
    const vc = it.member?.voice?.channel;
    if (!vc) {
      await it.reply({ content: 'Сначала зайди в голосовой канал.', flags: MessageFlags.Ephemeral }).catch(() => {});
      return;
    }
    followId = uid;
    followExplicit = true;
    emit({ evt: 'follow', user_id: uid, name: it.member?.displayName || it.user.username });
    await it.reply({ content: 'Иду. Буду ходить за тобой по каналам.', flags: MessageFlags.Ephemeral }).catch(() => {});
    joinChannel(it.guild, vc.id);
  } else if (it.commandName === 'leave') {
    await it.reply({ content: 'Вышел.', flags: MessageFlags.Ephemeral }).catch(() => {});
    leave(true);
  }
});

client.on(Events.Error, (e) => log('client', e.message));
client.on(Events.ShardDisconnect, () => emit({ evt: 'status', state: 'connecting' }));
client.on(Events.ShardResume, () => emit({ evt: 'status', state: current ? 'in_voice' : 'online' }));

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  switch (msg.cmd) {
    case 'skip':
      skipId = msg.user_id || null;
      break;
    case 'follow':
      followId = msg.user_id || null;
      followExplicit = Boolean(msg.user_id);
      if (followId) followTarget();
      break;
    case 'join':
      followTarget();
      break;
    case 'leave':
      leave(true);
      break;
    case 'members':
      emitMembers();
      break;
    case 'quit':
      shutdown();
      break;
  }
});
// родитель закрыл канал - выходим, а не висим сиротой
rl.on('close', () => shutdown());

// process.exit() при открытом stdin-пайпе на Windows роняет libuv
// (assert в async.c), поэтому закрываем всё и даём циклу событий кончиться.
let stopping = false;
function shutdown(code = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = code;
  try { leave(false); } catch {}
  try { rl.close(); } catch {}
  try { process.stdin.destroy(); } catch {}
  Promise.resolve(client.destroy()).catch(() => {});
  setTimeout(() => process.exit(code), 2000).unref();
}

if (TOKEN) {
  emit({ evt: 'status', state: 'connecting' });
  client.login(TOKEN).catch((e) => {
    const bad = /token/i.test(e.message) || e.code === 'TokenInvalid';
    emit({ evt: 'status', state: 'error', detail: bad ? 'bad-token' : e.message });
    shutdown(3);
  });
}
