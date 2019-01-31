/*
 * ***** BEGIN LICENSE BLOCK *****
 * Version: MIT
 *
 * Portions created by Alan Antonuk are Copyright (c) 2012-2013
 * Alan Antonuk. All Rights Reserved.
 *
 * Portions created by VMware are Copyright (c) 2007-2012 VMware, Inc.
 * All Rights Reserved.
 *
 * Portions created by Tony Garnock-Jones are Copyright (c) 2009-2010
 * VMware, Inc. and Tony Garnock-Jones. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * ***** END LICENSE BLOCK *****
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "common.hpp"
#ifdef WITH_SSL
#include "../thirdparty/rabbitmq-c/include/amqp_ssl_socket.h"
#endif

#include "../thirdparty/rabbitmq-c/include/amqp_tcp_socket.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef WINDOWS
#include "compat.h"
#endif

void die(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	exit(1);
}

void die_errno(int err, const char *fmt, ...) {
	va_list ap;

	if (err == 0) {
		return;
	}

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, ": %s\n", strerror(err));
	exit(1);
}

void die_amqp_error(int err, const char *fmt, ...) {
	va_list ap;

	if (err >= 0) {
		return;
	}

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, ": %s\n", amqp_error_string2(err));
	exit(1);
}

const char *amqp_server_exception_string(amqp_rpc_reply_t r) {
	int res;
	static char s[512];

	switch (r.reply.id) {
	case AMQP_CONNECTION_CLOSE_METHOD: {
		amqp_connection_close_t *m = (amqp_connection_close_t *)r.reply.decoded;
		res = snprintf(s, sizeof(s), "server connection error %d, message: %.*s",
		               m->reply_code, (int)m->reply_text.len,
		               (char *)m->reply_text.bytes);
		break;
	}

	case AMQP_CHANNEL_CLOSE_METHOD: {
		amqp_channel_close_t *m = (amqp_channel_close_t *)r.reply.decoded;
		res = snprintf(s, sizeof(s), "server channel error %d, message: %.*s",
		               m->reply_code, (int)m->reply_text.len,
		               (char *)m->reply_text.bytes);
		break;
	}

	default:
		res = snprintf(s, sizeof(s), "unknown server error, method id 0x%08X",
		               r.reply.id);
		break;
	}

	return res >= 0 ? s : NULL;
}

const char *amqp_rpc_reply_string(amqp_rpc_reply_t r) {
	switch (r.reply_type) {
	case AMQP_RESPONSE_NORMAL:
		return "normal response";

	case AMQP_RESPONSE_NONE:
		return "missing RPC reply type";

	case AMQP_RESPONSE_LIBRARY_EXCEPTION:
		return amqp_error_string2(r.library_error);

	case AMQP_RESPONSE_SERVER_EXCEPTION:
		return amqp_server_exception_string(r);

	default:
		abort();
	}
}

void die_rpc(amqp_rpc_reply_t r, const char *fmt, ...) {
	va_list ap;

	if (r.reply_type == AMQP_RESPONSE_NORMAL) {
		return;
	}

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, ": %s\n", amqp_rpc_reply_string(r));
	exit(1);
}

static int amqp_heartbeat = 0;
#ifdef WITH_SSL
static int amqp_ssl = 0;
static char *amqp_cacert = "/etc/ssl/certs/cacert.pem";
static char *amqp_key = NULL;
static char *amqp_cert = NULL;
#endif /* WITH_SSL */

const char *connect_options_title = "Connection options";

amqp_connection_state_t make_connection(struct amqp_connection_info* ci) {
	int status;
	amqp_socket_t *socket = NULL;
//	amqp_connection_state_t *conn =  (amqp_connection_state_t*) malloc(sizeof(amqp_connection_state_t));
	amqp_connection_state_t conn;
	conn = amqp_new_connection();
	if (ci->ssl) {
#ifdef WITH_SSL
		socket = amqp_ssl_socket_new(conn);
		if (!socket) {
			die("creating SSL/TLS socket");
		}
		if (amqp_cacert) {
			amqp_ssl_socket_set_cacert(socket, amqp_cacert);
		}
		if (amqp_key) {
			amqp_ssl_socket_set_key(socket, amqp_cert, amqp_key);
		}
#else
		die("librabbitmq was not built with SSL/TLS support");
#endif
	} else {
		socket = amqp_tcp_socket_new(conn);
		if (!socket) {
			die("creating TCP socket (out of memory)");
		}
	}
	status = amqp_socket_open(socket, ci->host, ci->port);
	if (status) {
		die("opening socket to %s:%d", ci->host, ci->port);
	}
	die_rpc(amqp_login(conn, ci->vhost, 1, AMQP_DEFAULT_FRAME_SIZE, amqp_heartbeat, AMQP_SASL_METHOD_PLAIN, ci->user, ci->password),
	        "logging in to AMQP server");
	if (!amqp_channel_open(conn, 1)) {
		die_rpc(amqp_get_rpc_reply(conn), "opening channel");
	}
	return conn;
}

void close_connection(amqp_connection_state_t conn) {
	int res;
	die_rpc(amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS), "closing channel");
	die_rpc(amqp_connection_close(conn, AMQP_REPLY_SUCCESS),
	        "closing connection");

	res = amqp_destroy_connection(conn);
	die_amqp_error(res, "closing connection");
}

amqp_bytes_t read_all(int fd) {
	size_t space = 4096;
	amqp_bytes_t bytes;

	bytes.bytes = malloc(space);
	bytes.len = 0;

	for (;;) {
		ssize_t res = read(fd, (char *)bytes.bytes + bytes.len, space - bytes.len);
		if (res == 0) {
			break;
		}

		if (res < 0) {
			if (errno == EINTR) {
				continue;
			}

			die_errno(errno, "reading");
		}

		bytes.len += res;
		if (bytes.len == space) {
			space *= 2;
			bytes.bytes = realloc(bytes.bytes, space);
		}
	}

	return bytes;
}

void write_all(int fd, amqp_bytes_t data) {
	while (data.len > 0) {
		ssize_t res = write(fd, data.bytes, data.len);
		if (res < 0) {
			die_errno(errno, "write");
		}

		data.len -= res;
		data.bytes = (char *)data.bytes + res;
	}
}

void copy_body(amqp_connection_state_t conn, int fd) {
	size_t body_remaining;
	amqp_frame_t frame;

	int res = amqp_simple_wait_frame(conn, &frame);
	die_amqp_error(res, "waiting for header frame");
	if (frame.frame_type != AMQP_FRAME_HEADER) {
		die("expected header, got frame type 0x%X", frame.frame_type);
	}

	body_remaining = frame.payload.properties.body_size;
	while (body_remaining) {
		res = amqp_simple_wait_frame(conn, &frame);
		die_amqp_error(res, "waiting for body frame");
		if (frame.frame_type != AMQP_FRAME_BODY) {
			die("expected body, got frame type 0x%X", frame.frame_type);
		}

		write_all(fd, frame.payload.body_fragment);
		body_remaining -= frame.payload.body_fragment.len;
	}
}

amqp_bytes_t cstring_bytes(const char *str) {
	return str ? amqp_cstring_bytes(str) : amqp_empty_bytes;
}
